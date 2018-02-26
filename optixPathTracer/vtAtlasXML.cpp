#include "vtAtlasXML.h"
#include <iostream>

using namespace pugi;
using namespace std;

vtAtlasXML::vtAtlasXML(string xml_filename)
{
	// Prepare xml document.
	xml_document doc;
	cout << "Loading texture atlas XML file...\n - Path: " << xml_filename << "\n - Result: ";
	xml_parse_result result = doc.load_file(xml_filename.c_str()); // Debug mode looks at build/<xml_filename>
	cout << result.description() << endl;

	// Parse texture atlas data into subtexture name, scale, and transform vectors of identical size.
	xml_node atlas = doc.child("TextureAtlas");
	m_filename = std::string(atlas.attribute("imagePath").value());
	m_texels_wide = atlas.attribute("width").as_int();
	m_texels_high = atlas.attribute("height").as_int();
	for (pugi::xml_node sprite : atlas.children("sprite"))
	{
		subtexture_names.push_back(sprite.attribute("n").as_string());
		subtexture_scale.push_back(pair<float, float>(sprite.attribute("w").as_float() / m_texels_wide, sprite.attribute("h").as_float() / m_texels_high));
		subtexture_translate.push_back(pair<float, float>(sprite.attribute("x").as_float() / m_texels_wide, sprite.attribute("y").as_float() / m_texels_high));
	}
}

vtAtlasXML::~vtAtlasXML() {}

std::string  vtAtlasXML::filename()    { return m_filename;    }
unsigned int vtAtlasXML::texels_wide() { return m_texels_wide; }
unsigned int vtAtlasXML::texels_high() { return m_texels_high; }

void vtAtlasXML::subtextureNameToST(std::string name, float& scale_x, float& scale_y, float& translate_x, float& translate_y)
{
	for (size_t i = 0; i < subtexture_names.size(); i++)
	{
		if (name.find(subtexture_names[i]) != std::string::npos)
		{
			scale_x = subtexture_scale[i].first;
			scale_y = subtexture_scale[i].second;
			translate_x = subtexture_translate[i].first;
			translate_y = 1.0f - subtexture_translate[i].second - subtexture_scale[i].second; // xml has inverted y-axis
			cout << "Texture name " << name << " matched atlas subtexture " << subtexture_names[i] << ".\n - Set ST: (" << scale_x << ", " << scale_y << ", " << translate_x << ", " << translate_y << ")." << endl;
		}
		else
		{
			// Commented out line below, because it's currently triggered for every non-diffuse texture map.
			//cout << "Texture name " << name << "was not matched to any subtexture name in the provided texture atlas xml. (Using default ST.)" << endl;
		}
	}
}
