#ifndef VT_TILE_READER_H
#define VT_TILE_READER_H

#include <string>
#include <IL/ilu.h>
#include "DevIL/devil_cpp_wrapper.h"

class vtTileReader
{
public:
	vtTileReader(const std::string folder_path, const std::string file_extension, ILubyte expected_bytes_per_texel, ILenum expected_image_format);
	~vtTileReader();
	void vtTileReader::GetTileImage(ilImage &tile, const unsigned int &tileMipID, const unsigned int &x, const unsigned int &y);
private:
	const std::string  m_folder_path;
	const std::string  m_file_extension;
	const ILubyte      m_bpp;
	const ILenum       m_format;
};

#endif // VT_TILE_READER_H
