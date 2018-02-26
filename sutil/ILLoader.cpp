
/*
*  Creates OptiX TextureSampler for any image supported by DevIL. Based on Nvidia's PPMLoader.
*/

#include "ILLoader.h"
#include <optixu/optixu_math_namespace.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "DevIL/devil_cpp_wrapper.h"

const unsigned int bytes_per_texel_in_source = 4;
const unsigned int bytes_per_texel_in_target = 4;
const ILenum devil_format = IL_RGBA;
const RTformat optix_format = RT_FORMAT_UNSIGNED_BYTE4;

using namespace optix;

//-----------------------------------------------------------------------------
//  
//  ILLoader class definition
//
//-----------------------------------------------------------------------------

ILLoader::ILLoader(const std::string& filename, const bool vflip)
	: m_nx(0u), m_ny(0u), m_raster(0)
{
	if (filename.empty()) return;

	// Check if file extension is supported by DevIL
	// TODO

	// Open file
	try {
		ilImage tex = ilImage(filename.c_str());

		// width, height
		m_nx = tex.Width();
		m_ny = tex.Height();

		m_raster = new(std::nothrow) unsigned char[m_nx*m_ny * bytes_per_texel_in_target];
		tex.Bind();
		ilCopyPixels(0, 0, 0, m_nx, m_ny, 1, devil_format, IL_UNSIGNED_BYTE, m_raster);

		if (vflip) {
			unsigned char *m_raster2 = new(std::nothrow)  unsigned char[m_nx*m_ny * bytes_per_texel_in_target];
			for (unsigned int y2 = m_ny - 1, y = 0; y<m_ny; y2--, y++) {
				for (unsigned int x = 0; x<m_nx * bytes_per_texel_in_target; x++)
					m_raster2[y2*m_nx * bytes_per_texel_in_target + x] = m_raster[y*m_nx * bytes_per_texel_in_target + x];
			}

			delete[] m_raster;
			m_raster = m_raster2;
		}
	}
	catch (...) {
		std::cerr << "ILLoader( '" << filename << "' ) failed to load" << std::endl;
		m_raster = 0;
	}
}


ILLoader::~ILLoader()
{
	if (m_raster) delete[] m_raster;
}


bool ILLoader::failed() const
{
	return m_raster == 0;
}


unsigned int ILLoader::width() const
{
	return m_nx;
}


unsigned int ILLoader::height() const
{
	return m_ny;
}


unsigned char* ILLoader::raster() const
{
	return m_raster;
}


//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

optix::TextureSampler ILLoader::loadTexture(optix::Context context,
	const float3& default_color,
	bool linearize_gamma)
{
	// lookup table for sRGB gamma linearization
	static unsigned char srgb2linear[256];
	// filling in a static lookup table for sRGB gamma linearization, standard formula for sRGB
	static bool srgb2linear_init = false;
	if (!srgb2linear_init) {
		srgb2linear_init = true;
		for (int i = 0; i < 256; i++) {
			float cs = i / 255.0f;
			if (cs <= 0.04045f)
				srgb2linear[i] = (unsigned char)(255.0f * cs / 12.92f + 0.5f);
			else
				srgb2linear[i] = (unsigned char)(255.0f * powf((cs + 0.055f) / 1.055f, 2.4f) + 0.5f);
		}
	}

	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = context->createTextureSampler();
	sampler->setWrapMode(0, RT_WRAP_REPEAT);
	sampler->setWrapMode(1, RT_WRAP_REPEAT);
	sampler->setWrapMode(2, RT_WRAP_REPEAT);
	sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	sampler->setMaxAnisotropy(1.0f);
	sampler->setMipLevelCount(1u);
	sampler->setArraySize(1u);

	if (failed()) {

		// Create buffer with single texel set to default_color
		optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, optix_format, 1u, 1u);
		unsigned char* buffer_data = static_cast<unsigned char*>(buffer->map());
		buffer_data[0] = (unsigned char)clamp((int)(default_color.x * 255.0f), 0, 255);
		buffer_data[1] = (unsigned char)clamp((int)(default_color.y * 255.0f), 0, 255);
		buffer_data[2] = (unsigned char)clamp((int)(default_color.z * 255.0f), 0, 255);
		if (bytes_per_texel_in_target > 3)
		{
			buffer_data[3] = 255;
		}
		buffer->unmap();

		sampler->setBuffer(0u, 0u, buffer);
		// Although it would be possible to use nearest filtering here, we chose linear
		// to be consistent with the textures that have been loaded from a file. This
		// allows OptiX to perform some optimizations.
		sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

		return sampler;
	}

	const unsigned int nx = width();
	const unsigned int ny = height();

	// Create buffer and populate with DevIL image data
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, optix_format, nx, ny);
	unsigned char* buffer_data = static_cast<unsigned char*>(buffer->map());

	for (unsigned int j = 0; j < ny; ++j) {
		for (unsigned int i = 0; i < nx; ++i) {

			unsigned int raster_index = (j*nx + i) * bytes_per_texel_in_source;
			unsigned int buffer_index = (j*nx + i) * bytes_per_texel_in_target;

			buffer_data[buffer_index + 0] = raster()[raster_index + 0];
			buffer_data[buffer_index + 1] = raster()[raster_index + 1];
			buffer_data[buffer_index + 2] = raster()[raster_index + 2];

			if (linearize_gamma) {
				buffer_data[buffer_index + 0] = srgb2linear[buffer_data[buffer_index + 0]];
				buffer_data[buffer_index + 1] = srgb2linear[buffer_data[buffer_index + 1]];
				buffer_data[buffer_index + 2] = srgb2linear[buffer_data[buffer_index + 2]];
			}

			if (bytes_per_texel_in_target > 3)
			{
				if (bytes_per_texel_in_source > 3)
				{
					buffer_data[buffer_index + 3] = raster()[raster_index + 3];
				}
				else
				{
					buffer_data[buffer_index + 3] = 255u;
				}
			}
		}
	}

	buffer->unmap();

	sampler->setBuffer(0u, 0u, buffer);
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

	return sampler;
}


//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

optix::TextureSampler loadILTexture(optix::Context context,
	const std::string& filename,
	const optix::float3& default_color)
{
	ILLoader ill(filename);
	return ill.loadTexture(context, default_color);
}


optix::Buffer loadILCubeBuffer(optix::Context context,
	const std::vector<std::string>& filenames)
{
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT | RT_BUFFER_CUBEMAP, optix_format);
	char* buffer_data = 0;
	for (size_t face = 0; face < filenames.size(); ++face) {
		// Read in HDR, set texture buffer to empty buffer if fails
		ILLoader ill(filenames[face]);
		if (ill.failed()) {
			return buffer;
		}

		const unsigned int nx = ill.width();
		const unsigned int ny = ill.height();

		if (face == 0) {
			buffer->setSize(nx, ny, filenames.size());
			buffer_data = static_cast<char*>(buffer->map());
		}
		else {
			buffer_data += nx * ny * sizeof(char) * bytes_per_texel_in_target;
		}

		for (unsigned int i = 0; i < nx; ++i) {
			for (unsigned int j = 0; j < ny; ++j) {
				unsigned int raster_index = ((j)*nx + i) * bytes_per_texel_in_target;
				unsigned int buffer_index = ((j)*nx + i) * bytes_per_texel_in_target;

				buffer_data[buffer_index + 0] = ill.raster()[raster_index + 0];
				buffer_data[buffer_index + 1] = ill.raster()[raster_index + 1];
				buffer_data[buffer_index + 2] = ill.raster()[raster_index + 2];
				if (bytes_per_texel_in_target > 3)
				{
					if (bytes_per_texel_in_source > 3)
					{
						buffer_data[buffer_index + 3] = ill.raster()[raster_index + 3];
					}
					else
					{
						buffer_data[buffer_index + 3] = 255u;
					}
				}
			}
		}

	}
	buffer->unmap();

	return buffer;
}
