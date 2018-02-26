#ifndef VT_MIPID_TEXTURE_H
#define VT_MIPID_TEXTURE_H

#include <optixu/optixpp_namespace.h>

class vtMipIDTexture
{
public:
	vtMipIDTexture(optix::Context context, const unsigned int dimensions);
	~vtMipIDTexture();

	optix::TextureSampler TextureSampler();
	optix::Buffer Buffer();
	const unsigned int Dimensions();
private:
	optix::TextureSampler m_sampler;
	optix::Buffer m_buffer;
	const unsigned int m_dimensions;
};

#endif // VT_MIPID_TEXTURE_H
