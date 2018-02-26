#include "vtTileReader.h"

vtTileReader::vtTileReader(const std::string folder_path, const std::string file_extension, ILubyte expected_bytes_per_texel, ILenum expected_image_format) :
	m_folder_path(folder_path),
	m_file_extension(file_extension),
	m_bpp(expected_bytes_per_texel),
	m_format(expected_image_format)
{}

vtTileReader::~vtTileReader() {}

void vtTileReader::GetTileImage(ilImage &tile, const unsigned int &tileMipID, const unsigned int &x, const unsigned int &y)
{
	ilImage tile_from_file((m_folder_path + "/tile_mipid_" + std::to_string(tileMipID) + "_x_" + std::to_string(x) + "_y_" + std::to_string(y) + m_file_extension).c_str());
	tile.TexImage(tile_from_file.Width(), tile_from_file.Height(), 1, m_bpp, m_format, IL_UNSIGNED_BYTE, NULL);
	ilOverlayImage(tile_from_file.GetId(), 0, 0, 0);
}
