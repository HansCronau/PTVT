#pragma once

#include <optixu/optixu_math_namespace.h>

static __host__ __device__ __inline__ int dimensionsForMipID(int mipID)
{
	return 1 << mipID;
}

static __host__ __device__ __inline__ int quick_log2(int n)
{
	int bits = 0;

	if (n > 0xffff) {
		n >>= 16;
		bits = 0x10;
	}

	if (n > 0xff) {
		n >>= 8;
		bits |= 0x8;
	}

	if (n > 0xf) {
		n >>= 4;
		bits |= 0x4;
	}

	if (n > 0x3) {
		n >>= 2;
		bits |= 0x2;
	}

	if (n > 0x1) {
		bits |= 0x1;
	}

	return bits;
}

static __host__ __device__ __inline__ int mipIDForDimensions(int dimensions)
{
    return quick_log2(dimensions);
}

static __host__ __device__ __inline__ int offsetForMipID(int mipID)
{
	int current_mip_offset = 0;
	for (int i = 0; i < mipID; i++)
	{
		int current_mip_dimensions = (int)pow(2.0f, (float)i);
		current_mip_offset += current_mip_dimensions*current_mip_dimensions;
	}
	return current_mip_offset;
}

static __host__ __device__ __inline__ int mipIDForOffset(int offset)
{
	int current_mip_offset = 0;
	for (int i = 0;; i++)
	{
		int current_mip_dimensions = dimensionsForMipID(i);
		int next_mip_offset = current_mip_offset + current_mip_dimensions*current_mip_dimensions;
		if (next_mip_offset > offset)
		{
			return i;
		}
		current_mip_offset = next_mip_offset;
	}
}

static __host__ __device__ __inline__ void tileIDForOffset(unsigned int offset, unsigned int& mipID, unsigned int& x, unsigned int& y)
{
	mipID = mipIDForOffset(offset);
	y = (offset - offsetForMipID(mipID)) / dimensionsForMipID(mipID);
	x = offset - offsetForMipID(mipID) - y * dimensionsForMipID(mipID);
}

// Assumes original image is mip level 0 and 1x1 image is mip id 0. Does not check if mip level is too high.
static __host__ __device__ __inline__ unsigned int mipLevelToMipID(unsigned int mipLevel, unsigned int textureDimensions) // Works for both mip levels and IDs expressed in tiles and in texels.
{
	unsigned int maxMipID = mipIDForDimensions(textureDimensions);
	return maxMipID - mipLevel;
}

// Assumes original image is mip level 0 and 1x1 image is mip id 0. Does not check if mip id is too high.
static __host__ __device__ __inline__ unsigned int mipIDToMipLevel(unsigned int mipID, unsigned int textureDimensions) // Works for both mip levels and IDs expressed in tiles and in texels.
{
	unsigned int maxMipID = mipIDForDimensions(textureDimensions);
	return maxMipID - mipID;
}

static __host__ __device__ __inline__ unsigned int texelMipIDToTileMipID(unsigned int texelMipID, unsigned int tileDimensionsInTexels)
{
	unsigned int texelMipIDForTileMipIDZero = mipIDForDimensions(tileDimensionsInTexels);
	return texelMipID - texelMipIDForTileMipIDZero;
}

static __host__ __device__ __inline__ unsigned int tileMipIDToTexelMipID(unsigned int tileMipID, unsigned int tileDimensionsInTexels)
{
	unsigned int texelMipIDForTileMipIDZero = mipIDForDimensions(tileDimensionsInTexels);
	return texelMipIDForTileMipIDZero + tileMipID;
}

static __host__ __device__ __inline__ unsigned int numberOfIndicesForDimensions(unsigned int dimensions)
{
	return offsetForMipID(mipIDForDimensions(dimensions) + 1);
}

static __host__ __device__ __inline__ unsigned int numberOfMipLevelsForDimensions(unsigned int dimensions)
{
	return mipIDForDimensions(dimensions) + 1; // +1 because mipID 0 (1x1) also counts.
}

// Returns float4 instead of float3, because OptiX context crashes on page table buffer of type RT_FORMAT_FLOAT3.
static __host__ __device__ __inline__ optix::float4 create_scale_and_bias(const unsigned int &virtual_tile_mip_id, const unsigned int &virtual_tile_x, const unsigned int &virtual_tile_y, const float &fallback_scale, const unsigned int &tile_texels_wide, const unsigned int &tile_border_texels_wide, const unsigned int &tile_pool_texels_wide, const float &physical_tile_x, const float &physical_tile_y)
{
	// Scale = Virtual Mip Level Width / Physical Tile Pool Width * Fallback Scale
	// Bias = Physical Coordinate - Scale * Virtual Coordinate

	// To minimise GPU calculations address translation from virtual to physical UV uses only one multiplication and one addition for respectively scale and bias.
	// Scaling virtual UV handles
	//   1. differences in size between the virtual texture mip level and the tile pool (including fall backs)
	//   2. tile borders.
	// Scaling yields an in itself meaningless coordinate. Bias is calculated after and based on scaling, to compensate for unintended offset of the coordinate.
	//	 bias = corner coordinate of payload in tile pool (excl. border) in UV [0,1] - physical tile pool UV-coordinate found by scaling virtual UV
	//	 corner coordinate of payload in tile pool (excl. border) in UV [0,1] = tile in pool corner coordinate (excl. border) in texels / pool texels wide
	//   tile in pool corner coordinate (excl. border) in texels = physical tile coordinate (CAN BE NONINTEGRAL!) * payload width + borders * border width
	// Note that physical for the last line "tile coordinate * tile width + border width" would introduce unwanted offset for fractional part of physical tile coordinate.

	optix::float4 scale_and_bias;
	const unsigned int virtual_tile_mip_tiles_wide = dimensionsForMipID(virtual_tile_mip_id);
	const unsigned int payload_texels_wide = (unsigned int)(tile_texels_wide - 2.0f * tile_border_texels_wide);

	// Scale
	scale_and_bias.x                                    = static_cast<float>(virtual_tile_mip_tiles_wide * payload_texels_wide) / tile_pool_texels_wide * fallback_scale;

	// Bias X
	const unsigned int nr_of_borders_before_payload_x   = static_cast<unsigned int>(physical_tile_x) * 2 + 1; // 2 borders per tile left of targeted tile + 1 border for the targeted tile
	const float physical_payload_lower_left_in_texels_x = physical_tile_x * payload_texels_wide + nr_of_borders_before_payload_x * tile_border_texels_wide;
	const float physical_payload_lower_left_in_uv_u     = physical_payload_lower_left_in_texels_x / tile_pool_texels_wide;
	scale_and_bias.y                                    = physical_payload_lower_left_in_uv_u - scale_and_bias.x * static_cast<float>(virtual_tile_x) / virtual_tile_mip_tiles_wide;

	// Bias Y
	const unsigned int nr_of_borders_before_payload_y   = static_cast<unsigned int>(physical_tile_y) * 2 + 1; // 2 borders per tile below targeted tile + 1 border for the targeted tile
	const float physical_payload_lower_left_in_texels_y = physical_tile_y * payload_texels_wide + nr_of_borders_before_payload_y * tile_border_texels_wide;
	const float physical_payload_lower_left_in_uv_v     = physical_payload_lower_left_in_texels_y / tile_pool_texels_wide;
	scale_and_bias.z                                    = physical_payload_lower_left_in_uv_v - scale_and_bias.x * static_cast<float>(virtual_tile_y) / virtual_tile_mip_tiles_wide;

	return scale_and_bias;
}

static __host__ __device__ __inline__ optix::float2 devirtualiseAddress(optix::float3 scaleAndBias, optix::float2 virtualUV)
{
	// Physical Coordinate = Scale * Virtual Coordinate + Bias
	optix::float2 physicalAddress;
	physicalAddress.x = scaleAndBias.x * virtualUV.x + scaleAndBias.y;
	physicalAddress.y = scaleAndBias.x * virtualUV.y + scaleAndBias.z;
	return physicalAddress;
}

static __host__ __device__ __inline__ float positive_modulo(float i, float n) {
	return fmod(fmod(i, n) + n, n);
}

static  __host__ __device__ __inline__ optix::float3 powf(optix::float3 a, float exp)
{
	return optix::make_float3(powf(a.x, exp), powf(a.y, exp), powf(a.z, exp));
}
