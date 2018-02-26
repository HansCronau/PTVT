#ifndef VT_TILE_CACHE_H
#define VT_TILE_CACHE_H

#include <optixu/optixpp_namespace.h>
#include "DevIL/devil_cpp_wrapper.h"

class vtTileCache
{
public:
	vtTileCache(optix::Context context, const unsigned int &cacheDimensions, const unsigned int &tileDimensions, const bool &debug_tile_borders);
	~vtTileCache();

	void UploadTile(ilImage &tile, const unsigned int &x, const unsigned int &y);
	void SetDebugTileBorders(const bool &onOff);

	optix::TextureSampler TextureSampler();
	optix::Buffer Buffer();
	const unsigned int DimensionsInTexels();
	const unsigned int DimensionsInTiles();
private:
	optix::TextureSampler m_sampler;
	optix::Buffer m_buffer;
	const unsigned int m_tile_dimensions_in_texels;
	const unsigned int m_cache_dimensions_in_texels;
	const unsigned int m_cache_dimensions_in_tiles;
	bool               m_draw_debug_tile_borders;
};

#endif // VT_TILE_CACHE_H
