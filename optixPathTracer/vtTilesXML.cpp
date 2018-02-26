#include "vtTilesXML.h"
#include <iostream>

using namespace pugi;
using namespace std;

vtTilesXML::vtTilesXML(std::string xml_filename)
{
	// Prepare xml document.
	xml_document doc;
	cout << "Loading tiles XML file...\n - Path: " << xml_filename << "\n - Result: ";
	xml_parse_result result = doc.load_file(xml_filename.c_str()); // Debug mode looks at build/<xml_filename>
	cout << result.description() << endl;

	// Get tile into.
	xml_node tile_info = doc.child("tile_info");
	m_tile_dimensions_in_texels = tile_info.attribute("dimensions_in_texels").as_int();
	m_tile_border_in_texels     = tile_info.attribute("border_in_texels").as_int();
	m_file_extension            = std::string(tile_info.attribute("file_extension").value());
}

vtTilesXML::~vtTilesXML() {}

unsigned int vtTilesXML::texels_wide()        { return m_tile_dimensions_in_texels; }
unsigned int vtTilesXML::border_texels_wide() { return m_tile_border_in_texels; }
std::string  vtTilesXML::file_extension()     { return m_file_extension; }

