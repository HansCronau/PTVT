#ifndef VT_PAGE_TABLE_H
#define VT_PAGE_TABLE_H

#include <optixu/optixpp_namespace.h>

class vtPageTable
{
public:
	vtPageTable(optix::Context context, const unsigned int dimensions);
	~vtPageTable();

	optix::TextureSampler TextureSampler();
	optix::Buffer Buffer();
	const unsigned int Dimensions();
private:
	optix::TextureSampler m_sampler;
	optix::Buffer m_buffer;
	const unsigned int m_dimensions;
};

#endif // VT_TILE_CACHE_H
