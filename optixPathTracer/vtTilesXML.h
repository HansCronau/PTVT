#ifndef VT_TILES_XML_H
#define VT_TILES_XML_H

#include <string>
#include <pugixml/pugixml.hpp>

class vtTilesXML
{
public:
	vtTilesXML(std::string xml_filename);
	~vtTilesXML();

	unsigned int texels_wide();
	unsigned int border_texels_wide();
	std::string  file_extension();

private:
	unsigned int   m_tile_dimensions_in_texels;
	unsigned int   m_tile_border_in_texels;
	std::string    m_file_extension;
};

#endif // VT_TILES_XML_H
