
/*
*  Creates OptiX TextureSampler for any image supported by DevIL. Based on Nvidia's PPMLoader.
*/

#pragma once

#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <string>
#include <iosfwd>

//-----------------------------------------------------------------------------
//
// Utility functions
//
//-----------------------------------------------------------------------------

// Creates a TextureSampler object for a given image file.  If filename is 
// empty or ILLoader fails, a 1x1 texture is created with the provided default
// texture color.
SUTILAPI optix::TextureSampler loadILTexture(optix::Context context,
	const std::string& il_filename,
	const optix::float3& default_color);

// Creates a Buffer object for the given cubemap files.
SUTILAPI optix::Buffer loadILCubeBuffer(optix::Context context,
	const std::vector<std::string>& filenames);

//-----------------------------------------------------------------------------
//
// ILLoader class declaration 
//
//-----------------------------------------------------------------------------

class ILLoader
{
public:
	SUTILAPI ILLoader(const std::string& filename, const bool vflip = false);
	SUTILAPI ~ILLoader();

	SUTILAPI optix::TextureSampler loadTexture(
		optix::Context context,
		const optix::float3& default_color,
		bool linearize_gamma = false
	);

	SUTILAPI bool           failed() const;
	SUTILAPI unsigned int   width() const;
	SUTILAPI unsigned int   height() const;
	SUTILAPI unsigned char* raster() const;

private:
	unsigned int   m_nx;
	unsigned int   m_ny;
	unsigned char* m_raster;
};
