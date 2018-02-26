/*
 * Path Traced Virtual Textures (PTVT)
 * Copyright 2018 Hans Cronau
 *
 * Demo based on the Optix SDK optixPathTracer sample,
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

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4, plane, , );
// Note: v1 and v2 are defined as offset1 and offset2 (vectors pointing from the anchor to adjacent corners of the parallelogram),
//       and divided by their original length squared (i.e. vector dotted with itself).
//       They are used to create ortogonal vector projections onto respectively offset1 and offset2 (for points defined relative to the anchor).
//       Math of projection vector a onto vector b: a_projected = dot(a,b)/lenth(b)^2*b
//       Division by the length of the vector squared is done because:
//         1. The dot product is divided by the length of b to find the scalar product (= length of the projection vector).
//            b normalised to unit length can be multiplied by this length to find the projection vector.
//            b is normalised by dividing it by its own length. This division is moved to front and combined with the other division by length.
//         2. This combination (the square of the length b) is easily calculated as the dot product of b with itself.
//            There's no need for any (computationally hard) square roots.
//       To empasise: v1 and v2 are NOT the edges of the parallelogram.
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, anchor, , );
rtDeclareVariable(float3, offset1, , );
rtDeclareVariable(float3, offset2, , );
rtDeclareVariable(int, lgt_instance, , ) = { 0 };

rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int, lgt_idx, attribute lgt_idx, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// Path differentials
rtDeclareVariable(float2, T_alpha, attribute T_alpha, );
rtDeclareVariable(float2, T_beta, attribute T_beta, );
rtDeclareVariable(float2, T_gamma, attribute T_gamma, );
rtDeclareVariable(float3, non_normalised_normal, attribute non_normalised_normal, );
rtDeclareVariable(float3, E_0, attribute E_0, );
rtDeclareVariable(float3, E_2, attribute E_2, );

// Debug - Barycentric coordinates
rtDeclareVariable(float3, P_alpha, attribute P_alpha, );
rtDeclareVariable(float3, P_gamma, attribute P_gamma, );

RT_PROGRAM void intersect(int primIdx)
{
    float3 n = make_float3(plane);
    float dt = dot(ray.direction, n);
    float t = (plane.w - dot(n, ray.origin)) / dt;
    if (t > ray.tmin && t < ray.tmax) {
        float3 p = ray.origin + ray.direction * t;
        float3 vi = p - anchor;
        float a1 = dot(v1, vi);
        if (a1 >= 0 && a1 <= 1){
            float a2 = dot(v2, vi);
            if (a2 >= 0 && a2 <= 1){
                if (rtPotentialIntersection(t)) {
					shading_normal = geometric_normal = n;

					// From this point on we'll treat the parallelogram as a (double) triangle.
					// This treatment makes it easier to apply the same differential calculations for both triangles and parallelograms.

					const float3 p0 = anchor;
					const float3 p1 = anchor + offset1;
					const float3 p2 = anchor + offset2;

					const float3 e0 = p1 - p0;
					const float3 e1 = p0 - p2;
                    // Note: e0 and e1 are named by OptiX convention. They correspond to E_0 and E_2 respectively.
					texcoord = make_float3(a1, a2, 0);

                    // Differentials
					// Add triangle-like texture coordinates.
					T_alpha = make_float2(0, 0);
					T_beta  = make_float2(1, 0);
					T_gamma = make_float2(0, 1);

                    non_normalised_normal = cross(e1, e0);
					E_0 = e0; // = p1 - p0
					E_2 = e1; // = p0 - p2

                    // Debug - Barycentric coordinates
                    P_alpha = p0;
                    P_gamma = p2;

                    lgt_idx = lgt_instance;
                    rtReportIntersection(0);
                }
            }
        }
    }
}

RT_PROGRAM void bounds(int, float result[6])
{
    // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
    const float3 tv1 = v1 / dot(v1, v1);
    const float3 tv2 = v2 / dot(v2, v2);
    const float3 p00 = anchor;
    const float3 p01 = anchor + tv1;
    const float3 p10 = anchor + tv2;
    const float3 p11 = anchor + tv1 + tv2;
    const float  area = length(cross(tv1, tv2));

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (area > 0.0f && !isinf(area)) {
        aabb->m_min = fminf(fminf(p00, p01), fminf(p10, p11));
        aabb->m_max = fmaxf(fmaxf(p00, p01), fmaxf(p10, p11));
    }
    else {
        aabb->invalidate();
    }
}
