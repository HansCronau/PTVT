/*
 * Path Traced Virtual Textures (PTVT)
 * Copyright 2018 Hans Cronau
 *
 * File based on the Optix SDK optixPathTracer sample,
 * Copyright 2016 Nvidia Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optixu/optixu_math_namespace.h>
#include "optixPathTracer.h"
#include "random.h"
#include "helpers.h"
#include <stdio.h>
#include "vtHelpers.h"
#include <sampleConfig.h>

using namespace optix;

struct PerRayData_pathtrace
{
    float3 result;
    float3 radiance;
    float3 attenuation;
    float3 origin;
    float3 direction;
    unsigned int seed;
    int depth;
    int countEmitted;
    int done;

    // Per ray support for virtual textures / path differentials:
    int within_fragment_path_id;     // required for virtual textures and path differentials
    float total_path_distance;       // required for virtual textures
    unsigned int differential_count; // required for differentials
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

// Path differentials
rtDeclareVariable(uint, differentialDepth, , );
rtBuffer<float3, 3>     positionDifferentials;
rtBuffer<float3, 3>     directionDifferentials;

// Custom render modes
rtDeclareVariable(uint, limit_to_one_bounce, , );
rtDeclareVariable(uint, debug_render_mode, , );

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(unsigned int, sqrt_num_samples, , );
rtDeclareVariable(unsigned int, rr_begin_depth, , );
rtDeclareVariable(unsigned int, pathtrace_ray_type, , );
rtDeclareVariable(unsigned int, pathtrace_shadow_ray_type, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;

RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();

    float2 inv_screen = 1.0f / make_float2(screen);
    float2 inv_screen_times_two = 2.0f * inv_screen;
    float2 pixel_screen = (make_float2(launch_index)) * inv_screen_times_two - 1.0f;
    // In the line above a pixel coordinate is created in screen space.
    // The launch_index is of type N^2 (incl. 0).
    // Is is converted to a screenspace in R^2 with the domain [-1.0, 1.0]^2.
    // Reason for the domain centered around 0.0 is that the camera's W vector points to the centre of the in-world screen (instead of for example to the top left).
    // The camera's U and V vectors in turn only span half the screen, from the centre to the edges.
    // pixel_screen.x and pixel_screen.y correspond respectively to x and y in Suykens2001 A.1 Pixel Sampling.

    float2 pixel_to_screen = inv_screen_times_two / sqrt_num_samples;
    // Scales pixel space to world space.
    // It is calculated here once so is doesn't have to be repeated in the loop below.
    // = (1.0 / sqrt_num_samples) * inv_screen_times_two

    unsigned int current_within_pixel_sample = sqrt_num_samples*sqrt_num_samples;
    // Counter for within-pixel samples.

    float3 result = make_float3(0.0f);

    unsigned int seed = tea<16>(screen.x*launch_index.y + launch_index.x, frame_number);
    do
    {
        //
        // Sample pixel using jittering
        //
        // Note: Samples are distributed evenly over pixel space. This, effectively, is supersampling.
        unsigned int current_within_pixel_sample_x = current_within_pixel_sample % sqrt_num_samples;
        unsigned int current_within_pixel_sample_y = current_within_pixel_sample / sqrt_num_samples;
        float2 current_within_pixel_sample_jittered = make_float2(
            current_within_pixel_sample_x - rnd(seed),
            current_within_pixel_sample_y - rnd(seed)
            );

        float2 point_in_screen = pixel_screen + current_within_pixel_sample_jittered * pixel_to_screen; // x * pix_w + pix_l (Suykens2001 A.1 Pixel Sampling)
        // In the line above I add a pixel's screen coordinate to the current within-pixel sample's coordinate, plus some random jitter.
        // The random values are within the range [0.0, 1.0] and subtracted from the absolute within-pixel sample's coordinates (domain [0, sqrt_num_samples)^2).
        // Combined they are scaled from (within-)pixel space to screen space (domain [-1, 1]^2) and added to the current pixel's screen space coordinate.
        // Notes on Igehy1999 and Suykens2001:
        // - The within-pixel coordinates in domain [0, 1]^2 are the unit random values x and y in Suykens2001.
        // - point_in_screen.x and point_in_screen.y are respectively called u and v in View + u*Right + v*Up.

        float3 ray_origin = eye;
        float3 ray_direction = normalize(point_in_screen.x*U + point_in_screen.y*V + W); // See Igehy1999 formula (4)

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.result = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;

        // Initialise VT data
        prd.within_fragment_path_id = current_within_pixel_sample - 1;
        prd.total_path_distance = 0.0f;

        // Initialise path differentials: create initial differentials
        prd.differential_count = 0;
        if (differentialDepth != 0) // TODO(HansCronau): Optimisation - We could assume differentialDepth is never < 1 to remove this if-statement.
        {
            const float3 dPdx = make_float3(0);
            const float3 dPdy = make_float3(0);
            // See Igehy1999 formula (8) and Suykens2001 A.1 Pixel Sampling
            const float3 dDdx = differential_generation_direction(ray_direction, U) * pixel_to_screen.x; // = dDdu * pix_w (Suykens2001 A.1 Pixel Sampling)
            const float3 dDdy = differential_generation_direction(ray_direction, V) * pixel_to_screen.y; // = dDdv * pix_h (Suykens2001 A.1 Pixel Sampling)

            // Determine index within differential buffers.
            const uint3 bufferIndexX = make_uint3(launch_index, prd.differential_count + 0);
            const uint3 bufferIndexY = make_uint3(launch_index, prd.differential_count + 1);
            positionDifferentials[bufferIndexX] = dPdx;
            positionDifferentials[bufferIndexY] = dPdy;
            directionDifferentials[bufferIndexX] = dDdx;
            directionDifferentials[bufferIndexY] = dDdy;

            prd.differential_count = 2u;
        }

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        for (;;)
        {
            Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
            rtTrace(top_object, ray, prd);

            // Debug quick rendering
            if (limit_to_one_bounce)
            {
                prd.result += prd.attenuation;
                break;
            }

            if (prd.done)
            {
                // We have hit the background or a luminaire
                prd.result += prd.radiance * prd.attenuation;
                break;
            }

            // Russian roulette termination 
            if (prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if (rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }

            prd.depth++;
            prd.result += prd.radiance * prd.attenuation;

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
        }

        result += prd.result;
        seed = prd.seed;
    } while (--current_within_pixel_sample);

    //
    // Update the output buffer
    //
    float3 pixel_color = result / (sqrt_num_samples*sqrt_num_samples);

    if (frame_number > 1)
    {
        float a = 1.0f / (float)frame_number;
        float3 old_color = make_float3(output_buffer[launch_index]);
        output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, a), 1.0f);
    }
    else
    {
        output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
    }
}


//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, emission_color, , );

RT_PROGRAM void diffuseEmitter()
{
    current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
    current_prd.done = true;
}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, diffuse_color, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

// Tile determination
rtBuffer<uint, 1>       tileID_to_frameID_buffer;
rtDeclareVariable(uint, frameID, , );
rtDeclareVariable(uint, tileID_path_depth, , );
rtDeclareVariable(uint, virt_tex_size, , );
rtDeclareVariable(uint, tile_size, , );
rtDeclareVariable(uint, tex_mip_id_of_lowest_mip_id_tile, , );
rtDeclareVariable(uint, max_tile_mip_id, , );
rtDeclareVariable(float, distance_to_onscreen_ratio, , );
rtDeclareVariable(float, footprint_scale, , );
rtDeclareVariable(float, footprint_bias, , );

// Path differential variables already defined scene wide (see above).

// From Material
rtTextureSampler<float4, 2> Kd_map;          // classic
rtDeclareVariable(int, texture_atlas, , );   // classic atlased
rtDeclareVariable(int, texture_atlas_0, , ); // classic atlased scaled mipmapped
rtDeclareVariable(int, texture_atlas_1, , ); // classic atlased scaled mipmapped
rtDeclareVariable(int, texture_atlas_2, , ); // classic atlased scaled mipmapped
rtDeclareVariable(int, texture_atlas_3, , ); // classic atlased scaled mipmapped
rtDeclareVariable(int, texture_atlas_4, , ); // classic atlased scaled mipmapped
rtDeclareVariable(int, texture_atlas_5, , ); // classic atlased scaled mipmapped
rtDeclareVariable(int, tile_pool, , );       // virtual atlased
rtDeclareVariable(int, page_table, , );
rtDeclareVariable(float4, texture_atlas_ST, , );
rtDeclareVariable(int, mipID_texture, , );

// Path differentials - Buffer for footprint calculations.
rtBuffer<float2, 3>     footprintDifferentials;

// Path differentials
rtDeclareVariable(float2, T_alpha, attribute T_alpha, ); // TODO(HansCronau): Optimisation - Can be removed by moving calculations to intersection program.
rtDeclareVariable(float2, T_beta, attribute T_beta, );
rtDeclareVariable(float2, T_gamma, attribute T_gamma, );
rtDeclareVariable(float3, non_normalised_normal, attribute non_normalised_normal, ); // TODO(HansCronau): Optimisation - Can be removed by moving calculations to intersection program.
rtDeclareVariable(float3, E_0, attribute E_0, );
rtDeclareVariable(float3, E_2, attribute E_2, );

// Debug - Barycentric coordinates
rtDeclareVariable(float3, P_alpha, attribute P_alpha, );
rtDeclareVariable(float3, P_gamma, attribute P_gamma, );

// Other variables from the intersection program
rtDeclareVariable(float3, texcoord, attribute texcoord, );


static __inline__ __device__ float3 mipToColour(unsigned int mipID)
{
    const int nr_of_colours = 16;
    const float3 colour_ramp[nr_of_colours] = {
        { .99f, 0.0f, 0.0f },
        { .99f, 0.35f, 0.0f },
        { .99f, 0.75f, 0.0f },
        { .99f, .99f, 0.0f },
        { 0.5f, .99f, 0.0f },
        { 0.11f, .99f, 0.0f },
        { 0.0f, .99f, 0.26f },
        { 0.0f, .99f, .99f },
        { 0.0f, 0.61f, .99f },
        { 0.0f, 0.26f, .99f },
        { 0.0f, 0.0f, .99f },
        { 0.4f, 0.0f, .99f },
        { 0.6f, 0.0f, .99f },
        { .99f, 0.0f, .99f },
        { .99f, 0.2f, .99f },
        { .99f, 0.6f, .99f }
    };
    const int ilod = min(mipID, nr_of_colours - 1);
    return colour_ramp[ilod];
}

RT_PROGRAM void diffuse()
{
    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
    float3 hitpoint = ray.origin + t_hit * ray.direction;


    //
    // TRANSFER existing position differentials. (No new differentials are created.)
    // Note: dPdy are included. For both dPdy and dPdx the __dx naming is used.
    //

    const float inv_DN = 1.0f / dot(ray.direction, ffnormal);
    const float3 normal_over_DN = ffnormal * inv_DN; // = ffnormal / dot(ray.direction, ffnormal)
    const float  test_t = dot(hitpoint, ffnormal) * inv_DN; // doesnt work. does this mean that igehy1999 formula 11 does not appy correctly (influencing 12 and 10)?

    for (unsigned int positionDifferential = 0; positionDifferential < current_prd.differential_count; positionDifferential++)
    {
        const uint3 bufferIndex = make_uint3(launch_index, positionDifferential);
        const float3 dPdx = positionDifferentials[bufferIndex];
        const float3 dDdx = directionDifferentials[bufferIndex];
        // Could replace below with function from helpers.h.
        const float dtdx = -dot(dPdx + t_hit * dDdx, normal_over_DN);      // See Igehy1999 formula 12.
        const float3 dPdx2 = (dPdx + t_hit * dDdx) + dtdx * ray.direction; // See Igehy1999 formula 10.
        positionDifferentials[bufferIndex] = dPdx2;
    }


    //
    // COMPUTE FOOTPRINT of path differentials.
    //

    // Calculate barycentric coordinate variables
    // Calculations inspired by Scratchapixel. https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
    // The order of computations was changed for optimisation and easy application to differentials.
    // Correctness L_beta and L_gamma can be checked with barycentric render mode.
    const float  inv_denom = 1.0 / dot(non_normalised_normal, non_normalised_normal);
    const float3 n_inv_denom = non_normalised_normal * inv_denom;
    const float3 L_beta = cross(n_inv_denom, E_2);
    const float3 L_gamma = cross(n_inv_denom, E_0);
    // TODO(HansCronau): Optimisation - Move these calculations to intersection program for optimisation (speed and possibly memory).

    // Calculate footprint differentials (Igehy1999)
    float2 footprint_T = make_float2(0.f);
    for (unsigned int positionDifferential = 0; positionDifferential < current_prd.differential_count; positionDifferential++)
    {
        const uint3 bufferIndex = make_uint3(launch_index, positionDifferential);
        const float3 dPdx_world = positionDifferentials[bufferIndex];

        const float3 dPdx_object = rtTransformVector(RT_WORLD_TO_OBJECT, dPdx_world);

        // Calculate barycentric coordinates.
        const float beta = dot(L_beta, dPdx_object);
        const float gamma = dot(L_gamma, dPdx_object);

        // Calculate differential as one would calculate ordinary UV coordinates,
        // but make values relative by subtracting a corner vector's position.
        const float2 dTdx = beta*(T_beta - T_alpha) + gamma*(T_gamma - T_alpha); // + (1.0f - beta - gamma)*(T_alpha - T_alpha) is left out because it equals 0.
        // Vector dTdx represents a difference, but can be both positive or negative.
        // Make sure all dTdx are expressed positively. (All vectors should point up in Fig. 3(b), Suykens2001.)
        const float2 dTdx_pos = dTdx.y > 0.0f ? dTdx : -dTdx;

        // Store footprint differential for processing (see next step).
        footprintDifferentials[bufferIndex] = dTdx_pos;
        // Add footprint differential to sum of all footprints.
        footprint_T += dTdx_pos;
    }

    // Process footprint differentials
    const float2 footprint_PT = make_float2(footprint_T.y, -footprint_T.x); // PT is any vector perpendicular to T.
    float2 footprint_DA = make_float2(0);
    for (unsigned int positionDifferential = 0; positionDifferential < current_prd.differential_count; positionDifferential++)
    {
        const uint3 bufferIndex = make_uint3(launch_index, positionDifferential);
        const float2 dTdx = footprintDifferentials[bufferIndex];
        if (dot(footprint_PT, dTdx) > 0)
        {
            footprint_DA += dTdx;
        }
    }
    float2 footprint_DB = footprint_T - footprint_DA;


    //
    // CHOOSE DELTA
    //

    const unsigned int num_samples = sqrt_num_samples*sqrt_num_samples;                          // = N (Suykens2001)
    const float suykens_delta = 1.0f / powf(num_samples, 1.0f / current_prd.differential_count); // Is $1/\sqrt[M]{N}$ (Suykens2001)
    const float delta = 2.0f * suykens_delta;                                                    // Because U and V both span only half the respective width and height of the screen. (See inv_screen_times_two)

    // Scale footprint from UV space to texture's texel space (correct for atlas scale) and apply delta.
    const uint virt_tex_size_in_tiles = virt_tex_size / tile_size;
    const float2 texture_atlas_S = make_float2(texture_atlas_ST.x, texture_atlas_ST.y);

    // Combine scaling effects: delta (Suykens2001) * subtexture to atlas (scales down differentials) * custom scaling (from host device)
    const float2 combined_footprint_scale = delta * texture_atlas_S * footprint_scale;
    // NOTE(HansCronau): Optimisation possible if footprints are only used for virtual textures: move multiplication by tile_size to line above.

    // Calculate texture differentials from footprint vectors.
    const float2 dTdx_A = combined_footprint_scale * footprint_DA + footprint_bias * normalize(footprint_DA);
    const float2 dTdx_B = combined_footprint_scale * footprint_DB + footprint_bias * normalize(footprint_DB);
    // NOTE(HansCronau): Optimisation possible if bias is 0: remove bias calculations.

    // Upscale differentials when sampling from textures representing tile data per texel (i.e. mipID texture and page table texture).
    const float2 dTiledx_A = tile_size * dTdx_A; // = texel mipID to tile mipID * differential
    const float2 dTiledx_B = tile_size * dTdx_B;

    //
    // WRITE TO TILEID BUFFER (VIRTUAL TEXTURE TILE DETERMINATION)
    //

    // Scale and translate UV coordinates from subtexture space to atlas texture space. Modulo is to support tiling textures.
    float atlased_u = positive_modulo(texcoord.x, 1.0f) * texture_atlas_ST.x + texture_atlas_ST.z;
    float atlased_v = positive_modulo(texcoord.y, 1.0f) * texture_atlas_ST.y + texture_atlas_ST.w;

    // If current path hasn't had too many bounces, we have room in our buffer to record hit tileIDs.
    uint tile_mipID_calc, tile_mipID_tex, tile_mipID, tile_x, tile_y;
    if (current_prd.depth < tileID_path_depth) {
        // Determine index within tileID buffer to write to.
        uint3 bufferIdx = make_uint3(launch_index, current_prd.within_fragment_path_id * tileID_path_depth + current_prd.depth); // nog checken voor randgevallen

        // Determine mipID. (MipID is an inverse mipmap level so that 2^mipID yields mip level's dimensions.)

        // Option A: Calculate mipID from differentials. (Chajdas2010, OpenGL 4.6 8.14.1)
        const float mipCalcHack = 5.5f; // TODO(HansCronau): Remove hack. Don't know why it is required. Expected same results as when sampling from mipID texture.
        const int tile_mip_level = mipCalcHack + log2(max(length(dTiledx_A), length(dTiledx_B))); // Note: rounding down mip level by cast to int increases mip resolution.
        tile_mipID_calc = max_tile_mip_id - clamp(tile_mip_level, 0, static_cast<int>(max_tile_mip_id));

        // Option B: Sample mipID from mipID texture using differentials.
        // This implicitly caps the mipID to the max mip ID.
        tile_mipID_tex = rtTex2DGrad<uint>(mipID_texture, atlased_u, atlased_v, dTiledx_A, dTiledx_B);

        // Option C: Calculate mipID based on path distance.
        // Keep track of the total distance traveled by the path.
        //current_prd.total_path_distance += t_hit;
        //uint tex_mipID = ceil(log2f(virt_tex_size / (current_prd.total_path_distance * distance_to_onscreen_ratio)));
        //tile_mipID = clamp(tex_mipID - tex_mip_id_of_lowest_mip_id_tile), 0, max_tile_mip_id);

        tile_mipID = tile_mipID_calc;

        // Calculate remaining TileID x and y.
        int mipWidth = dimensionsForMipID(tile_mipID);
        tile_x = atlased_u * mipWidth;
        tile_y = atlased_v * mipWidth;

        // Find the corresponding tile index in the tileID_to_frameID_buffer.
        uint tileID_index = offsetForMipID(tile_mipID) + mipWidth * tile_y + tile_x;

        // Store current frameID at that index.
        tileID_to_frameID_buffer[tileID_index] = frameID;
    }
    else
    {
        // Warning: values below are invalid when 1 or higher. (Your graphics driver may crash.)
        tile_x = 0;
        tile_y = 0;
        tile_mipID = 0;
    }


    //
    // SHADE
    // Note: f/pdf = 1 since we are perfectly importance sampling lambertian with cosine density.
    //

    // NOTE(HansCronau): For debugging purposes here comes a big nasty switch statement.

    // Variable names used in multiple switch statement cases must be defined outside of it:
    float3 scaleAndBias;
    float3 colour;

    switch (debug_render_mode)
    {
    case 0:
        // Render diffuse single colour
        colour = make_float3(t_hit / 10000.0f);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 1:
        // Render barycentric coordinates
        const float3 P_intersection = rtTransformVector(RT_WORLD_TO_OBJECT, hitpoint);
        const float beta = dot(L_beta, P_intersection - P_gamma);
        const float gamma = dot(L_gamma, P_intersection - P_alpha);
        colour = make_float3(1 - beta - gamma, beta, gamma);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 2:
        // Render interpolated UV
        colour = fminf(texcoord, make_float3(.99, .99, .99));
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 3:
        // Render diffuse with classic texture
        colour = make_float3(tex2D(Kd_map, texcoord.x, texcoord.y)); // Standard texture sampling.
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 4:
        // Render atlased interpolated UV
        colour = make_float3(atlased_u, atlased_v, 0);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 5:
        // Render diffuse with classic texture atlas
        colour = make_float3(rtTex2D<float4>(texture_atlas, atlased_u, atlased_v));
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 6:
        // Render diffuse with classic, scaled texture atlas
        colour = make_float3(rtTex2D<float4>(texture_atlas_5, atlased_u, atlased_v));
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 7:
        // Render diffuse with classic, scaled, and mipmapped texture atlas
        if (tile_mipID == 0)
        {
            colour = make_float3(rtTex2D<float4>(texture_atlas_0, atlased_u, atlased_v));
        }
        else if (tile_mipID == 1)
        {
            colour = make_float3(rtTex2D<float4>(texture_atlas_1, atlased_u, atlased_v));
        }
        else if (tile_mipID == 2)
        {
            colour = make_float3(rtTex2D<float4>(texture_atlas_2, atlased_u, atlased_v));
        }
        else if (tile_mipID == 3)
        {
            colour = make_float3(rtTex2D<float4>(texture_atlas_3, atlased_u, atlased_v));
        }
        else if (tile_mipID == 4)
        {
            colour = make_float3(rtTex2D<float4>(texture_atlas_4, atlased_u, atlased_v));
        }
        else
        {
            colour = make_float3(rtTex2D<float4>(texture_atlas_5, atlased_u, atlased_v));
        }
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 8:
        // Render differential footprint
        colour = make_float3(footprint_T.x, -footprint_T.x, footprint_T.y);
        colour = clamp(colour, 0, .99);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 9:
        // Render atlased differential footprint
        colour = make_float3(texture_atlas_ST.x * footprint_T.x, -texture_atlas_ST.x * footprint_T.x, texture_atlas_ST.y * footprint_T.y);
        colour = clamp(colour, 0, .99);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 10:
        // Render atlased differential footprint A component
        colour = make_float3(texture_atlas_ST.x * dTdx_A.x, texture_atlas_ST.x * -dTdx_A.x, texture_atlas_ST.x * dTdx_A.y);
        colour = clamp(colour, 0, .99);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 11:
        // Render atlased differential footprint B component
        colour = make_float3(texture_atlas_ST.y * dTdx_B.x, texture_atlas_ST.y * -dTdx_B.x, texture_atlas_ST.y * dTdx_B.y);
        colour = clamp(colour, 0, .99);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 12:
        // Render mipID colours sampling mipID from texture
        colour = mipToColour(tile_mipID_tex);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 13:
        // Render mipID colours calculating mipID from differentials
        colour = mipToColour(tile_mipID_calc);
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 14:
        // Render TileID
        if (tile_mipID == 0)
        {
            // MipID 0 corresponds to the colour black (0,0,0), which does not render well. Use grey instead.
            colour = make_float3(.01, .01, .01);
        }
        else
        {
            colour = make_float3(
                (float)tile_mipID / mipIDForDimensions(virt_tex_size_in_tiles),
                (float)tile_x / virt_tex_size_in_tiles,
                (float)tile_y / virt_tex_size_in_tiles
                );
        }
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 15:
        // Render scaleAndBias.
        //scaleAndBias = make_float3(rtTex2DLod<float4>(page_table, texcoord.x, texcoord.y, 0.0)); // Sample from page table texture.
        scaleAndBias = make_float3(rtTex2DGrad<float4>(page_table, atlased_u, atlased_v, dTiledx_A, dTiledx_B)); // Sample from page table texture.
        colour.x = scaleAndBias.x / (virt_tex_size / tile_size);
        colour.y = scaleAndBias.y;
        colour.z = scaleAndBias.z;
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 16:
        // Render tile pool directly.
        colour = make_float3(rtTex2D<float4>(tile_pool, atlased_u, atlased_v)); // Direct tile pool sampling.
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    case 17:
    default:
        // Render diffuse with virtual texture.
        scaleAndBias = make_float3(rtTex2DGrad<float4>(page_table, atlased_u, atlased_v, dTiledx_A, dTiledx_B)); // Sample from page table texture.
        float2 physicalAddress = devirtualiseAddress(scaleAndBias, make_float2(atlased_u, atlased_v)); // Translate virtual to physical address.
        colour = make_float3(rtTex2D<float4>(tile_pool, physicalAddress.x, physicalAddress.y)); // Virtual texture sampling.
        current_prd.attenuation = current_prd.attenuation * colour;
        break;
    }

    current_prd.countEmitted = false;


    //
    // SCATTER
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //

    current_prd.origin = hitpoint;

    // Create some random variables between 0 and 1.
    float z1 = rnd(current_prd.seed);
    float z2 = rnd(current_prd.seed);

    // Create an orthonormal basis (ONB).
    optix::Onb onb(ffnormal);

    // If using differentials and room left in differential buffer:
    if (current_prd.differential_count < differentialDepth * 2)
    {
        // Sample a point and differentials on a cosine weighted hemisphere.
        float3 p, dpdz1, dpdz2;
        cosine_sample_hemisphere_incl_differentials(z1, z2, p, dpdz1, dpdz2);

        // Project the point on the hemisphere and the differentials onto the orthonormal basis.
        current_prd.direction = onb.m_tangent * p.x + onb.m_binormal * p.y + ffnormal * p.z;
        const float3 dDdx = onb.m_tangent * dpdz1.x + onb.m_binormal * dpdz1.y + ffnormal * dpdz1.z;
        const float3 dDdy = onb.m_tangent * dpdz2.x + onb.m_binormal * dpdz2.y + ffnormal * dpdz2.z;

        // Create new position differentials.
        const float3 dPdx = make_float3(0);
        const float3 dPdy = make_float3(0);

        // Determine index within differential buffers.
        const uint3 differentialBufferIndexX = make_uint3(launch_index, current_prd.differential_count + 0);
        const uint3 differentialBufferIndexY = make_uint3(launch_index, current_prd.differential_count + 1);

        // Add new differentials to differential buffers.
        positionDifferentials[differentialBufferIndexX] = dPdx;
        positionDifferentials[differentialBufferIndexY] = dPdy;
        directionDifferentials[differentialBufferIndexX] = dDdx;
        directionDifferentials[differentialBufferIndexY] = dDdy;
        current_prd.differential_count += 2u;
    }
    else // Classic diffuse scatter without calculating differentials.
    {
        // Sample a point and differentials on a cosine weighted hemisphere.
        float3 p;
        cosine_sample_hemisphere(z1, z2, p);

        // Project the point on the hemisphere onto the orthonormal basis.
        // Equivalent to current_prd.direction = tangent * p.x + binormal * p.y + ffnormal * p.z;
        // Binormal is graphics lingo for bitangent.
        onb.inverse_transform(p);
        current_prd.direction = p;
    }


    //
    // NEXT EVENT ESTIMATION
    // Compute direct lighting.
    //

    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    for (int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L = normalize(light_pos - hitpoint);
        const float  nDl = dot(ffnormal, L);
        const float  LnDl = dot(light.normal, L);

        // cast shadow ray
        if (nDl > 0.0f && LnDl > 0.0f)
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
            rtTrace(top_object, shadow_ray, shadow_prd);

            if (!shadow_prd.inShadow)
            {
                const float A = length(cross(light.v1, light.v2));
                // convert area based pdf to solid angle
                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                result += light.emission * weight;
            }
        }
    }

    current_prd.radiance = result;
}


// This closest hit program is an adapted copy-paste of the above diffuse() program.
// Code was removed so that it runs only the (non-virtual) classic, scaled texture atlas render mode.
RT_PROGRAM void atlased_diffuse()
{
    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
    float3 hitpoint = ray.origin + t_hit * ray.direction;

    //
    // SHADE
    // Note: f/pdf = 1 since we are perfectly importance sampling lambertian with cosine density.
    //

    // Scale and translate UV coordinates from subtexture space to atlas texture space. Modulo is to support tiling textures.
    float atlased_u = positive_modulo(texcoord.x, 1.0f) * texture_atlas_ST.x + texture_atlas_ST.z;
    float atlased_v = positive_modulo(texcoord.y, 1.0f) * texture_atlas_ST.y + texture_atlas_ST.w;

    // Render diffuse with classic texture atlas
    float3 colour = make_float3(rtTex2D<float4>(texture_atlas, atlased_u, atlased_v));
    current_prd.attenuation = current_prd.attenuation * colour;
    current_prd.countEmitted = false;


    //
    // SCATTER
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //

    current_prd.origin = hitpoint;

    // Create some random variables between 0 and 1.
    float z1 = rnd(current_prd.seed);
    float z2 = rnd(current_prd.seed);

    // Create an orthonormal basis (ONB).
    optix::Onb onb(ffnormal);

    // Classic diffuse scatter without calculating differentials.
    // Sample a point and differentials on a cosine weighted hemisphere.
    float3 p;
    cosine_sample_hemisphere(z1, z2, p);

    // Project the point on the hemisphere onto the orthonormal basis.
    // Equivalent to current_prd.direction = tangent * p.x + binormal * p.y + ffnormal * p.z;
    // Binormal is graphics lingo for bitangent.
    onb.inverse_transform(p);
    current_prd.direction = p;


    //
    // NEXT EVENT ESTIMATION
    // Compute direct lighting.
    //

    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    for (int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L = normalize(light_pos - hitpoint);
        const float  nDl = dot(ffnormal, L);
        const float  LnDl = dot(light.normal, L);

        // cast shadow ray
        if (nDl > 0.0f && LnDl > 0.0f)
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
            rtTrace(top_object, shadow_ray, shadow_prd);

            if (!shadow_prd.inShadow)
            {
                const float A = length(cross(light.v1, light.v2));
                // convert area based pdf to solid angle
                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                result += light.emission * weight;
            }
        }
    }

    current_prd.radiance = result;
}


// This closest hit program is an adapted copy-paste of the above diffuse() program.
// Code was removed so that it runs only the virtual texture render mode.
RT_PROGRAM void virtual_diffuse()
{
    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
    float3 hitpoint = ray.origin + t_hit * ray.direction;


    //
    // TRANSFER existing position differentials. (No new differentials are created.)
    // Note: dPdy are included. For both dPdy and dPdx the __dx naming is used.
    //

    const float inv_DN = 1.0f / dot(ray.direction, ffnormal);
    const float3 normal_over_DN = ffnormal * inv_DN; // = ffnormal / dot(ray.direction, ffnormal)
    const float  test_t = dot(hitpoint, ffnormal) * inv_DN; // doesnt work. does this mean that igehy1999 formula 11 does not appy correctly (influencing 12 and 10)?

    for (unsigned int positionDifferential = 0; positionDifferential < current_prd.differential_count; positionDifferential++)
    {
        const uint3 bufferIndex = make_uint3(launch_index, positionDifferential);
        const float3 dPdx = positionDifferentials[bufferIndex];
        const float3 dDdx = directionDifferentials[bufferIndex];
        // Could replace below with function from helpers.h.
        const float dtdx = -dot(dPdx + t_hit * dDdx, normal_over_DN);      // See Igehy1999 formula 12.
        const float3 dPdx2 = (dPdx + t_hit * dDdx) + dtdx * ray.direction; // See Igehy1999 formula 10.
        positionDifferentials[bufferIndex] = dPdx2;
    }


    //
    // COMPUTE FOOTPRINT of path differentials.
    //

    // Calculate barycentric coordinate variables
    // Calculations inspired by Scratchapixel. https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
    // The order of computations was changed for optimisation and easy application to differentials.
    // Correctness L_beta and L_gamma can be checked with barycentric render mode.
    const float  inv_denom = 1.0 / dot(non_normalised_normal, non_normalised_normal);
    const float3 n_inv_denom = non_normalised_normal * inv_denom;
    const float3 L_beta = cross(n_inv_denom, E_2);
    const float3 L_gamma = cross(n_inv_denom, E_0);
    // TODO(HansCronau): Optimisation - Move these calculations to intersection program for optimisation (speed and possibly memory).

    // Calculate footprint differentials (Igehy1999)
    float2 footprint_T = make_float2(0.f);
    for (unsigned int positionDifferential = 0; positionDifferential < current_prd.differential_count; positionDifferential++)
    {
        const uint3 bufferIndex = make_uint3(launch_index, positionDifferential);
        const float3 dPdx_world = positionDifferentials[bufferIndex];

        const float3 dPdx_object = rtTransformVector(RT_WORLD_TO_OBJECT, dPdx_world);

        // Calculate barycentric coordinates.
        const float beta = dot(L_beta, dPdx_object);
        const float gamma = dot(L_gamma, dPdx_object);

        // Calculate differential as one would calculate ordinary UV coordinates,
        // but make values relative by subtracting a corner vector's position.
        const float2 dTdx = beta*(T_beta - T_alpha) + gamma*(T_gamma - T_alpha); // + (1.0f - beta - gamma)*(T_alpha - T_alpha) is left out because it equals 0.
        // Vector dTdx represents a difference, but can be both positive or negative.
        // Make sure all dTdx are expressed positively. (All vectors should point up in Fig. 3(b), Suykens2001.)
        const float2 dTdx_pos = dTdx.y > 0.0f ? dTdx : -dTdx;

        // Store footprint differential for processing (see next step).
        footprintDifferentials[bufferIndex] = dTdx_pos;
        // Add footprint differential to sum of all footprints.
        footprint_T += dTdx_pos;
    }

    // Process footprint differentials
    const float2 footprint_PT = make_float2(footprint_T.y, -footprint_T.x); // PT is any vector perpendicular to T.
    float2 footprint_DA = make_float2(0);
    for (unsigned int positionDifferential = 0; positionDifferential < current_prd.differential_count; positionDifferential++)
    {
        const uint3 bufferIndex = make_uint3(launch_index, positionDifferential);
        const float2 dTdx = footprintDifferentials[bufferIndex];
        if (dot(footprint_PT, dTdx) > 0)
        {
            footprint_DA += dTdx;
        }
    }
    float2 footprint_DB = footprint_T - footprint_DA;


    //
    // CHOOSE DELTA
    //

    const unsigned int num_samples = sqrt_num_samples*sqrt_num_samples;                          // = N (Suykens2001)
    const float suykens_delta = 1.0f / powf(num_samples, 1.0f / current_prd.differential_count); // Is $1/\sqrt[M]{N}$ (Suykens2001)
    const float delta = 2.0f * suykens_delta;                                                    // Because U and V both span only half the respective width and height of the screen. (See inv_screen_times_two)

    // Scale footprint from UV space to texture's texel space (correct for atlas scale) and apply delta.
    const uint virt_tex_size_in_tiles = virt_tex_size / tile_size;
    const float2 texture_atlas_S = make_float2(texture_atlas_ST.x, texture_atlas_ST.y);

    // Combine scaling effects: delta (Suykens2001) * subtexture to atlas (scales down differentials) * custom scaling (from host device)
    const float2 combined_footprint_scale = delta * texture_atlas_S * footprint_scale;
    // NOTE(HansCronau): Optimisation possible if footprints are only used for virtual textures: move multiplication by tile_size to line above.

    // Calculate texture differentials from footprint vectors.
    const float2 dTdx_A = combined_footprint_scale * footprint_DA + footprint_bias * normalize(footprint_DA);
    const float2 dTdx_B = combined_footprint_scale * footprint_DB + footprint_bias * normalize(footprint_DB);
    // NOTE(HansCronau): Optimisation possible if bias is 0: remove bias calculations.

    // Upscale differentials when sampling from textures representing tile data per texel (i.e. mipID texture and page table texture).
    const float2 dTiledx_A = tile_size * dTdx_A; // = texel mipID to tile mipID * differential
    const float2 dTiledx_B = tile_size * dTdx_B;

    //
    // WRITE TO TILEID BUFFER (VIRTUAL TEXTURE TILE DETERMINATION)
    //

    // Scale and translate UV coordinates from subtexture space to atlas texture space. Modulo is to support tiling textures.
    float atlased_u = positive_modulo(texcoord.x, 1.0f) * texture_atlas_ST.x + texture_atlas_ST.z;
    float atlased_v = positive_modulo(texcoord.y, 1.0f) * texture_atlas_ST.y + texture_atlas_ST.w;

    // If current path hasn't had too many bounces, we have room in our buffer to record hit tileIDs.
    uint tile_mipID_calc, tile_mipID_tex, tile_mipID, tile_x, tile_y;
    if (current_prd.depth < tileID_path_depth) {
        // Determine index within tileID buffer to write to.
        uint3 bufferIdx = make_uint3(launch_index, current_prd.within_fragment_path_id * tileID_path_depth + current_prd.depth); // nog checken voor randgevallen

        // Determine mipID. (MipID is an inverse mipmap level so that 2^mipID yields mip level's dimensions.)

        // Option A: Calculate mipID from differentials. (Chajdas2010, OpenGL 4.6 8.14.1)
        const float mipCalcHack = 5.5f; // TODO(HansCronau): Remove hack. Don't know why it is required. Expected same results as when sampling from mipID texture.
        const int tile_mip_level = mipCalcHack + log2(max(length(dTiledx_A), length(dTiledx_B))); // Note: rounding down mip level by cast to int increases mip resolution.
        tile_mipID_calc = max_tile_mip_id - clamp(tile_mip_level, 0, static_cast<int>(max_tile_mip_id));

        // Option B: Sample mipID from mipID texture using differentials.
        // This implicitly caps the mipID to the max mip ID.
        //tile_mipID_tex = rtTex2DGrad<uint>(mipID_texture, atlased_u, atlased_v, dTiledx_A, dTiledx_B);

        // Option C: Calculate mipID based on path distance.
        // Keep track of the total distance traveled by the path.
        //current_prd.total_path_distance += t_hit;
        //uint tex_mipID = ceil(log2f(virt_tex_size / (current_prd.total_path_distance * distance_to_onscreen_ratio)));
        //tile_mipID = clamp(tex_mipID - tex_mip_id_of_lowest_mip_id_tile), 0, max_tile_mip_id);

        tile_mipID = tile_mipID_calc;

        // Calculate remaining TileID x and y.
        int mipWidth = dimensionsForMipID(tile_mipID);
        tile_x = atlased_u * mipWidth;
        tile_y = atlased_v * mipWidth;

        // Find the corresponding tile index in the tileID_to_frameID_buffer.
        uint tileID_index = offsetForMipID(tile_mipID) + mipWidth * tile_y + tile_x;

        // Store current frameID at that index.
        tileID_to_frameID_buffer[tileID_index] = frameID;
    }
    else
    {
        // Warning: values below are invalid when 1 or higher. (Your graphics driver may crash.)
        tile_x = 0;
        tile_y = 0;
        tile_mipID = 0;
    }


    //
    // SHADE
    // Note: f/pdf = 1 since we are perfectly importance sampling lambertian with cosine density.
    //

    // Render diffuse with virtual texture.
    float3 scaleAndBias = make_float3(rtTex2DGrad<float4>(page_table, atlased_u, atlased_v, dTiledx_A, dTiledx_B)); // Sample from page table texture.
    float2 physicalAddress = devirtualiseAddress(scaleAndBias, make_float2(atlased_u, atlased_v)); // Translate virtual to physical address.
    float3 colour = make_float3(rtTex2D<float4>(tile_pool, physicalAddress.x, physicalAddress.y)); // Virtual texture sampling.
    current_prd.attenuation = current_prd.attenuation * colour;
    current_prd.countEmitted = false;


    //
    // SCATTER
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //

    current_prd.origin = hitpoint;

    // Create some random variables between 0 and 1.
    float z1 = rnd(current_prd.seed);
    float z2 = rnd(current_prd.seed);

    // Create an orthonormal basis (ONB).
    optix::Onb onb(ffnormal);

    // If using differentials and room left in differential buffer:
    if (current_prd.differential_count < differentialDepth * 2)
    {
        // Sample a point and differentials on a cosine weighted hemisphere.
        float3 p, dpdz1, dpdz2;
        cosine_sample_hemisphere_incl_differentials(z1, z2, p, dpdz1, dpdz2);

        // Project the point on the hemisphere and the differentials onto the orthonormal basis.
        current_prd.direction = onb.m_tangent * p.x + onb.m_binormal * p.y + ffnormal * p.z;
        const float3 dDdx = onb.m_tangent * dpdz1.x + onb.m_binormal * dpdz1.y + ffnormal * dpdz1.z;
        const float3 dDdy = onb.m_tangent * dpdz2.x + onb.m_binormal * dpdz2.y + ffnormal * dpdz2.z;

        // Create new position differentials.
        const float3 dPdx = make_float3(0);
        const float3 dPdy = make_float3(0);

        // Determine index within differential buffers.
        const uint3 differentialBufferIndexX = make_uint3(launch_index, current_prd.differential_count + 0);
        const uint3 differentialBufferIndexY = make_uint3(launch_index, current_prd.differential_count + 1);

        // Add new differentials to differential buffers.
        positionDifferentials[differentialBufferIndexX] = dPdx;
        positionDifferentials[differentialBufferIndexY] = dPdy;
        directionDifferentials[differentialBufferIndexX] = dDdx;
        directionDifferentials[differentialBufferIndexY] = dDdy;
        current_prd.differential_count += 2u;
    }
    else // Classic diffuse scatter without calculating differentials.
    {
        // Sample a point and differentials on a cosine weighted hemisphere.
        float3 p;
        cosine_sample_hemisphere(z1, z2, p);

        // Project the point on the hemisphere onto the orthonormal basis.
        // Equivalent to current_prd.direction = tangent * p.x + binormal * p.y + ffnormal * p.z;
        // Binormal is graphics lingo for bitangent.
        onb.inverse_transform(p);
        current_prd.direction = p;
    }


    //
    // NEXT EVENT ESTIMATION
    // Compute direct lighting.
    //

    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    for (int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L = normalize(light_pos - hitpoint);
        const float  nDl = dot(ffnormal, L);
        const float  LnDl = dot(light.normal, L);

        // cast shadow ray
        if (nDl > 0.0f && LnDl > 0.0f)
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
            rtTrace(top_object, shadow_ray, shadow_prd);

            if (!shadow_prd.inShadow)
            {
                const float A = length(cross(light.v1, light.v2));
                // convert area based pdf to solid angle
                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                result += light.emission * weight;
            }
        }
    }

    current_prd.radiance = result;
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
    current_prd_shadow.inShadow = true;
    rtTerminateRay();
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
#ifdef VT_DEBUG_OPTIX_PRINT_ENABLED
    rtPrintf("Exception!\n");
#endif
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss()
{
#ifdef VT_DEBUG_OPTIX_PRINT_ENABLED
    rtPrintf("Miss.\n");
#endif
    current_prd.radiance = bg_color;
    current_prd.done = true;
}
