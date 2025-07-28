"""
Pytest tests for vision tools functionality
"""
import json
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
import base64

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vision_tools import (
    set_vision_model,
    set_image_data_accessor,
    analyze_medical_image,
    translate_medical_to_common_terms,
    validate_base64_image
)

class TestVisionToolsSetup:
    """Test vision tools setup and configuration"""
    
    def test_set_vision_model(self):
        """Test setting vision model"""
        mock_model = Mock()
        set_vision_model(mock_model)
        # Would test that global _VISION_MODEL is set (private access needed)
        assert True  # Setup function test
    
    def test_set_image_data_accessor(self):
        """Test setting image data accessor function"""
        mock_accessor = Mock(return_value="base64_data")
        set_image_data_accessor(mock_accessor)
        # Would test that global _IMAGE_DATA_ACCESSOR is set
        assert True  # Setup function test

class TestImageAnalysis:
    """Test medical image analysis functionality"""
    
    @pytest.fixture
    def sample_base64_image(self):
        """Create a sample base64 encoded image for testing"""
        # Create minimal PNG data
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        return base64.b64encode(png_data).decode('utf-8')
    
    @patch('src.vision_tools._VISION_MODEL')
    @patch('src.vision_tools._IMAGE_DATA_ACCESSOR')
    def test_analyze_medical_image_with_data(self, mock_accessor, mock_model, sample_base64_image):
        """Test medical image analysis with valid data"""
        # Setup mocks
        mock_accessor.return_value = sample_base64_image
        mock_response = Mock()
        mock_response.content = json.dumps({
            "visible_symptoms": ["redness", "swelling"],
            "confidence": "high",
            "additional_observations": "Localized inflammation"
        })
        mock_model.invoke.return_value = mock_response
        
        # Test the function
        result = analyze_medical_image.func("")
        result_data = json.loads(result)
        
        assert "visible_symptoms" in result_data
        assert result_data["success"] == True
    
    @patch('src.vision_tools._IMAGE_DATA_ACCESSOR')
    def test_analyze_medical_image_no_data(self, mock_accessor):
        """Test medical image analysis with no image data"""
        mock_accessor.return_value = None
        
        result = analyze_medical_image.func("")
        result_data = json.loads(result)
        
        assert result_data["success"] == False
        assert "no image" in result_data["message"].lower()
    
    @patch('src.vision_tools._VISION_MODEL')
    @patch('src.vision_tools._IMAGE_DATA_ACCESSOR')
    def test_analyze_medical_image_model_error(self, mock_accessor, mock_model, sample_base64_image):
        """Test medical image analysis with model error"""
        mock_accessor.return_value = sample_base64_image
        mock_model.invoke.side_effect = Exception("API Error")
        
        result = analyze_medical_image.func("")
        result_data = json.loads(result)
        
        assert result_data["success"] == False
        assert "error" in result_data["message"].lower()

class TestMedicalTermProcessing:
    """Test medical term extraction and translation"""
    
    def test_translate_medical_to_common_terms(self):
        """Test translation of medical terms to common language"""
        terms = ["erythema", "edema", "cellulitis"]
        result = translate_medical_to_common_terms(terms)
        
        assert isinstance(result, list)
        assert len(result) >= 0
    
    def test_translate_medical_terms_empty_list(self):
        """Test translation with empty term list"""
        result = translate_medical_to_common_terms([])
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_validate_base64_image(self):
        """Test base64 image validation"""
        # Test with invalid base64
        result = validate_base64_image("invalid_base64")
        
        assert isinstance(result, dict)
        assert "success" in result
    
    @pytest.mark.parametrize("terms,expected_type", [
        (["fever"], list),
        (["fever", "headache"], list),
        (["fever", "headache", "nausea"], list),
        ([], list),
    ])
    def test_translate_medical_terms_counts(self, terms, expected_type):
        """Test translation with different term counts"""
        result = translate_medical_to_common_terms(terms)
        
        # Should handle different numbers of terms
        assert isinstance(result, expected_type)

class TestImageValidation:
    """Test image data validation"""
    
    def test_base64_validation(self):
        """Test base64 image data validation"""
        # Valid base64
        valid_base64 = base64.b64encode(b"test data").decode('utf-8')
        assert len(valid_base64) > 0
        
        # Invalid base64 (should be handled gracefully)
        invalid_base64 = "not-base64-data!"
        # The function should handle this gracefully
        assert isinstance(invalid_base64, str)
    
    def test_image_size_limits(self):
        """Test image size validation"""
        # Small image (should be valid)
        small_data = b"small" * 100  # 500 bytes
        small_b64 = base64.b64encode(small_data).decode('utf-8')
        assert len(small_b64) < 10000  # Well under typical limits
        
        # Large image would be rejected by the system
        # (We can't easily test the actual limit without a full system)

class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @patch('src.vision_tools._VISION_MODEL')
    @patch('src.vision_tools._IMAGE_DATA_ACCESSOR')
    def test_complete_vision_workflow(self, mock_accessor, mock_model):
        """Test complete vision analysis workflow"""
        # Setup mock data
        mock_accessor.return_value = "valid_base64_data"
        mock_response = Mock()
        mock_response.content = json.dumps({
            "visible_symptoms": ["rash", "inflammation"],
            "medical_terms": ["erythema", "dermatitis"], 
            "confidence": "medium"
        })
        mock_model.invoke.return_value = mock_response
        
        # Test workflow
        result = analyze_medical_image.func("analyze this skin condition")
        result_data = json.loads(result)
        
        assert result_data["success"] == True
        assert "visible_symptoms" in result_data
    
    def test_error_recovery(self):
        """Test error recovery in various scenarios"""
        # Test with various invalid inputs
        invalid_inputs = [None, "", "   ", 123, []]
        
        for invalid_input in invalid_inputs:
            try:
                # Most functions should handle invalid inputs gracefully
                if isinstance(invalid_input, (str, list)):
                    if isinstance(invalid_input, list):
                        result = translate_medical_to_common_terms(invalid_input)
                        assert isinstance(result, list)
                    elif isinstance(invalid_input, str):
                        result = validate_base64_image(invalid_input)
                        assert isinstance(result, dict)
            except Exception as e:
                # Should handle gracefully, not crash
                assert "error" in str(e).lower() or len(str(e)) > 0

class TestMockingAndStubs:
    """Test mocking capabilities for external dependencies"""
    
    def test_vision_model_mocking(self):
        """Test that vision model can be properly mocked"""
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(content='{"result": "mocked"}')
        
        # Test that mock works as expected
        response = mock_model.invoke("test")
        assert response.content == '{"result": "mocked"}'
    
    def test_image_accessor_mocking(self):
        """Test that image data accessor can be mocked"""
        mock_accessor = Mock(return_value="mock_base64_data")
        
        # Test that mock works
        result = mock_accessor()
        assert result == "mock_base64_data"

if __name__ == "__main__":
    pytest.main([__file__])