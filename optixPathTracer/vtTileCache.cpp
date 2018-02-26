#include "vtTileCache.h"
#include <iostream>
#include "DevIL/devil_cpp_wrapper.h"

const unsigned int bytes_per_texel = 4;
const RTformat format = RT_FORMAT_UNSIGNED_BYTE4;

// Both cacheDimensions and tileDimensions are in pixels.
vtTileCache::vtTileCache(optix::Context context, const unsigned int &cacheDimensions, const unsigned int &tileDimensions, const bool &debug_tile_borders) :
m_tile_dimensions_in_texels(tileDimensions),
m_cache_dimensions_in_texels(cacheDimensions),
m_cache_dimensions_in_tiles(cacheDimensions / tileDimensions),
m_draw_debug_tile_borders(debug_tile_borders)
{
	// Create a texture sampler and populate it with default values.
	m_sampler = context->createTextureSampler();
	m_sampler->setWrapMode(0, RT_WRAP_REPEAT);
	m_sampler->setWrapMode(1, RT_WRAP_REPEAT);
	m_sampler->setWrapMode(2, RT_WRAP_REPEAT);
	m_sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	m_sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	m_sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	m_sampler->setMaxAnisotropy(1.0f);
	m_sampler->setArraySize(1u);

	// Create a buffer and populate it "black"
	m_buffer = context->createBuffer(RT_BUFFER_INPUT, format, m_cache_dimensions_in_texels, m_cache_dimensions_in_texels);
	unsigned char* buffer_data = static_cast<unsigned char*>(m_buffer->map());

	for (unsigned int i = 0; i < m_cache_dimensions_in_texels; ++i) {
		for (unsigned int j = 0; j < m_cache_dimensions_in_texels; ++j) {

			unsigned int buffer_index = (i * m_cache_dimensions_in_texels + j) * bytes_per_texel;

			for (unsigned int i = 0; i < bytes_per_texel; i++)
			{
				buffer_data[buffer_index + i] = 0;
			}
		}
	}

	m_buffer->unmap();
	m_sampler->setBuffer(0u, 0u, m_buffer);
}

vtTileCache::~vtTileCache() {}

// Note: Assumes that nr of bytes/channels in tileData match that of the tile pool.
void vtTileCache::UploadTile(ilImage &tile, const unsigned int &x, const unsigned int &y)
{

	if (x > m_cache_dimensions_in_tiles || y > m_cache_dimensions_in_tiles)
	{
		std::cout << "Tile data was to be uploaded to a tile coordinate outside of the tile cache.";
		exit(13);
	}

	ILubyte * const tile_data = tile.GetData();
	unsigned char* buffer_data = static_cast<unsigned char*>(m_buffer->map());

	for (unsigned int i = 0; i < m_tile_dimensions_in_texels; ++i) {
		for (unsigned int j = 0; j < m_tile_dimensions_in_texels; ++j) {

			unsigned int tile_index = (i * m_tile_dimensions_in_texels + j) * bytes_per_texel;
			unsigned int buffer_index = ((y * m_tile_dimensions_in_texels + i) * m_cache_dimensions_in_texels + (x * m_tile_dimensions_in_texels + j)) * bytes_per_texel;

			if (buffer_index > m_cache_dimensions_in_texels * m_cache_dimensions_in_texels * bytes_per_texel)
			{
				std::cout << "Tile data was to be uploaded to a texel coordinate outside of the tile cache.";
				exit(14);
			}
			if (tile_index > m_cache_dimensions_in_texels * m_cache_dimensions_in_texels * bytes_per_texel)
			{
				std::cout << "Tile data was read from a texel coordinate outside of the tile.";
				exit(15);
			}

			for (unsigned int k = 0; k < bytes_per_texel; k++)
			{
				buffer_data[buffer_index + k] = tile_data[tile_index + k];
			}
			
			if (m_draw_debug_tile_borders)
			{
				unsigned int const vt_debug_tile_border_texels_wide = 8;
				if (   i <  vt_debug_tile_border_texels_wide
					|| j <  vt_debug_tile_border_texels_wide
					|| i >= m_tile_dimensions_in_texels - vt_debug_tile_border_texels_wide
					|| j >= m_tile_dimensions_in_texels - vt_debug_tile_border_texels_wide
				)
				{
					buffer_data[buffer_index + 0] = 255;
					for (unsigned int l = 1; l < bytes_per_texel; l++)
					{
						buffer_data[buffer_index + l] = 0;
					}
				}
			}
		}
	}

	m_buffer->unmap();
}

void vtTileCache::SetDebugTileBorders(const bool &onOff)
{
	m_draw_debug_tile_borders = onOff;
}

optix::TextureSampler vtTileCache::TextureSampler()  { return m_sampler; }
optix::Buffer vtTileCache::Buffer()                  { return m_buffer; }
const unsigned int vtTileCache::DimensionsInTexels() { return m_cache_dimensions_in_texels; }
const unsigned int vtTileCache::DimensionsInTiles()  { return m_cache_dimensions_in_tiles; }
