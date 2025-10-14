"""
Unit tests for O*NET Mapper
"""

import pytest
from pathlib import Path
import pandas as pd
import uuid

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'backend' / 'etl'))

from onet_mapper import ONetMapper, generate_deterministic_uuid


class TestONetMapper:
    """Test cases for ONetMapper class"""
    
    @pytest.fixture
    def mapper(self, tmp_path):
        """Create an ONetMapper instance with temporary directory"""
        return ONetMapper(data_dir=str(tmp_path))
    
    def test_initialization(self, mapper):
        """Test that mapper initializes correctly"""
        assert mapper.data_dir is not None
        assert mapper.mapped_dir is not None
        assert mapper.similarity_threshold == 85
    
    def test_deterministic_uuid_generation(self):
        """Test that UUIDs are deterministic"""
        uuid1 = generate_deterministic_uuid('knowledge', 'Python Programming')
        uuid2 = generate_deterministic_uuid('knowledge', 'Python Programming')
        uuid3 = generate_deterministic_uuid('skill', 'Python Programming')
        
        # Same entity type and name should produce same UUID
        assert uuid1 == uuid2
        
        # Different entity type should produce different UUID
        assert uuid1 != uuid3
        
        # Should be valid UUID format
        assert uuid.UUID(uuid1)
    
    def test_get_type_from_hierarchy(self, mapper):
        """Test hierarchy-based type mapping"""
        mapper.hierarchy_mappings = {
            '2.A.1': 'Technical',
            '2.B.1': 'Analytical'
        }
        
        assert mapper.get_type_from_hierarchy('2.A.1.a') == 'Technical'
        assert mapper.get_type_from_hierarchy('2.B.1.b') == 'Analytical'
        assert mapper.get_type_from_hierarchy('9.X.X') == 'General'
    
    def test_get_level_from_rating(self, mapper):
        """Test level dimension mapping from ratings"""
        assert mapper.get_level_from_rating(7.0, 'K') == 'Expert'
        assert mapper.get_level_from_rating(5.0, 'S') == 'Advanced'
        assert mapper.get_level_from_rating(3.5, 'A') == 'Intermediate'
        assert mapper.get_level_from_rating(2.0, 'K') == 'Basic'
        assert mapper.get_level_from_rating(None, 'S') == 'Basic'
    
    def test_entity_deduplication(self, mapper):
        """Test that duplicate entities are properly deduplicated"""
        # This would test the deduplication logic in map_knowledge_entities
        # For now, just verify the method exists
        assert hasattr(mapper, 'map_knowledge_entities')
        assert hasattr(mapper, 'map_skill_entities')