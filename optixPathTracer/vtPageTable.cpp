#include "vtHelpers.h"
#include "vtPageTable.h"
#include <iostream>
#include "DevIL/devil_cpp_wrapper.h"

const unsigned int elements_per_texel = 4; // ST-scale, S-bias, T-bias
const RTformat format = RT_FORMAT_FLOAT4;
// Also note that m_buffer is mapped to float*.

// A float for each is 4 bytes each (= 32 bits). 16 bits each is required for good results with page tables of over 1024*1024 pages. [VanWaveren2012]
// This means float is overkill. Could make buffer smaller by using other format. Also by cutting page table in two (see [VanWaveren2012]).

// Both cacheDimensions and tileDimensions are in pixels.
vtPageTable::vtPageTable(optix::Context context, const unsigned int dimensions) : m_dimensions(dimensions)
{
	// Create a texture sampler.
	m_sampler = context->createTextureSampler();

	m_sampler->setWrapMode(0, RT_WRAP_REPEAT); // Wrap in all dimensions. (Shouldn't matter.)
	m_sampler->setWrapMode(1, RT_WRAP_REPEAT);
	m_sampler->setWrapMode(2, RT_WRAP_REPEAT);
	m_sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES); // UV coordinates from 0 to 1.
	m_sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Read the elements as they are. (Don't normalise them.)
	m_sampler->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NEAREST); // Read the EXACT value (address data) of the nearest tile representing texel. (No filtering between texels/tile addresses.)
	m_sampler->setMaxAnisotropy(1.0f); // No anisotropic filtering for the page table.
	m_sampler->setMipLevelClamp(0.0f, 1000.0f); // Have page table texture support up to 1000.0 mipmaps.
	m_sampler->setArraySize(1u); // No texture arrays required.

	// Create a mipmapped buffer.
	unsigned int nrOfMipLevels = numberOfMipLevelsForDimensions(dimensions);
	m_buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, format, m_dimensions, m_dimensions, nrOfMipLevels);
	
	// Populate the buffer with nothingness.
	for (unsigned int mipID = 0; mipID < nrOfMipLevels; mipID++)
	{
		float* buffer_data = static_cast<float*>(m_buffer->map(mipIDToMipLevel(mipID, m_dimensions))); // Uses float!
		unsigned int mipDimension = dimensionsForMipID(mipID);

		for (unsigned int i = 0; i < mipDimension; ++i) {
			for (unsigned int j = 0; j < mipDimension; ++j) {

				unsigned int buffer_index = (i * mipDimension + j) * elements_per_texel;

				for (unsigned int i = 0; i < elements_per_texel; i++)
				{
					buffer_data[buffer_index + i] = 0;
				}
			}
		}

		m_buffer->unmap(mipIDToMipLevel(mipID, m_dimensions));
	}

	m_sampler->setBuffer(0u, 0u, m_buffer);
}

vtPageTable::~vtPageTable() {}

optix::TextureSampler vtPageTable::TextureSampler()  { return m_sampler; }
optix::Buffer vtPageTable::Buffer()                  { return m_buffer; }
const unsigned int vtPageTable::Dimensions()         { return m_dimensions; }
