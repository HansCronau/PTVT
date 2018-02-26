#ifndef VT_ATLAS_XML_H
#define VT_ATLAS_XML_H

#include <string>
#include <vector>
#include "pugixml/pugixml.hpp"

class vtAtlasXML
{
public:
	vtAtlasXML(std::string xml_filename);
	~vtAtlasXML();

	std::string  vtAtlasXML::filename();
	unsigned int vtAtlasXML::texels_wide();
	unsigned int vtAtlasXML::texels_high();

	void subtextureNameToST(std::string name, float& scale_x, float& scale_y, float& translate_x, float& translate_y);

	std::vector<std::string> subtexture_names;
	std::vector<std::pair<float, float>> subtexture_scale;
	std::vector<std::pair<float, float>> subtexture_translate;

private:
	std::string  m_filename;
	unsigned int m_texels_wide;
	unsigned int m_texels_high;
};

#endif // VT_ATLAS_XML_H
