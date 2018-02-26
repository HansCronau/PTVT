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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

// This is to be plugged into an RTgeometry object to represent
// a triangle mesh with a vertex buffer of triangle soup (triangle list)
// with an interleaved position, normal, texturecoordinate layout.

rtBuffer<float3> vertex_buffer;     
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   index_buffer;
rtBuffer<int>    material_buffer;

rtDeclareVariable(float3, texcoord,         attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(float3, back_hit_point,   attribute back_hit_point, ); 
rtDeclareVariable(float3, front_hit_point,  attribute front_hit_point, ); 

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

// Legenda variables:
// corners triangle:                                   p0,    p1,   p2
// opposing edges:                                            e1,   e0
// corresponding barycentric coordinates intersection: alpha, beta, gamma
// cartesian coordinates intersection (object space):          i
static __device__ bool intersect_triangle_branchless(const Ray&    ray,
                                                     const float3& p0,
                                                     const float3& p1,
													 const float3& p2,
													 const float3& e0,
													 const float3& e1,
                                                           float3& n,
                                                           float&  t,
                                                           float&  beta,
                                                           float&  gamma)
{
  n  = cross( e1, e0 );

  const float3 m = ( 1.0f / dot( n, ray.direction ) ) * ( p0 - ray.origin );
  const float3 i = cross(ray.direction, m);

  beta  = dot( i, e1 );
  gamma = dot( i, e0 );
  t     = dot( n, m );

  return ( (t<ray.tmax) & (t>ray.tmin) & (beta>=0.0f) & (gamma>=0.0f) & (beta+gamma<=1) );
}

RT_PROGRAM void mesh_intersect(int primIdx)
{
	const int3 v_idx = index_buffer[primIdx];

	const float3 p0 = vertex_buffer[ v_idx.x ];
	const float3 p1 = vertex_buffer[ v_idx.y ];
	const float3 p2 = vertex_buffer[ v_idx.z ];

	// Will be used for triangle intersection and differential calculation.
	const float3 e0 = p1 - p0;
    const float3 e1 = p0 - p2;
    // Note: e0 and e1 are named by OptiX convention. They correspond to E_0 and E_2 respectively.
	
	// Intersect ray with triangle.
	float3 n;
	float  t, beta, gamma;
	if (intersect_triangle_branchless(ray, p0, p1, p2, e0, e1, n, t, beta, gamma)) {

		if(  rtPotentialIntersection( t ) ) {

			// Normals
			geometric_normal = normalize( n );
			if( normal_buffer.size() == 0 ) {
			shading_normal = geometric_normal; 
			} else {
			float3 n0 = normal_buffer[ v_idx.x ];
			float3 n1 = normal_buffer[ v_idx.y ];
			float3 n2 = normal_buffer[ v_idx.z ];
			shading_normal = normalize( n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );
			}

			// Differentials - Texture coordinates
			if (texcoord_buffer.size() == 0) {
				T_alpha  = make_float2( 0.0f, 0.0f );
				T_beta   = make_float2( 0.0f, 0.0f );
				T_gamma  = make_float2( 0.0f, 0.0f );
				texcoord = make_float3( 0.0f, 0.0f, 0.0f ); // OptiX code has UV-coords in R3.
			}
			else {
				T_alpha  = texcoord_buffer[ v_idx.x ];
				T_beta   = texcoord_buffer[ v_idx.y ];
				T_gamma  = texcoord_buffer[ v_idx.z ];
				texcoord = make_float3(T_beta*beta + T_gamma*gamma + T_alpha*(1.0f - beta - gamma));
			}

			// Differentials - Edges
            non_normalised_normal = n;
			E_0 = e0; // = p1 - p0
			E_2 = e1; // = p0 - p2

            // Debug - Barycentric coordinates
            P_alpha = p0;
            P_gamma = p2;

			rtReportIntersection(material_buffer[primIdx]);
		}
	}
}

RT_PROGRAM void mesh_bounds (int primIdx, float result[6])
{
  const int3 v_idx = index_buffer[primIdx];

  const float3 v0   = vertex_buffer[ v_idx.x ];
  const float3 v1   = vertex_buffer[ v_idx.y ];
  const float3 v2   = vertex_buffer[ v_idx.z ];
  const float  area = length(cross(v1-v0, v2-v0));

  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( v0, v1), v2 );
    aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
  } else {
    aabb->invalidate();
  }
}

