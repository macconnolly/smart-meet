#!/usr/bin/env python3
"""
Simple test to verify the basic pipeline components work.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_models():
    """Test that basic models work."""
    print("🧪 Testing models...")
    try:
        from src.models.entities import Memory, MemoryType, ContentType
        memory = Memory(
            id="test-1",
            meeting_id="meeting-1", 
            content="This is a test memory",
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.DECISION
        )
        print(f"✅ Memory created: {memory.content[:30]}...")
        return True
    except Exception as e:
        print(f"❌ Models failed: {e}")
        return False

def test_temporal_extraction():
    """Test temporal dimension extraction."""
    print("🧪 Testing temporal extraction...")
    try:
        from src.extraction.dimensions.temporal_extractor import TemporalDimensionExtractor
        
        extractor = TemporalDimensionExtractor()
        
        # Test urgent content
        result = extractor.extract("This is urgent! We need to complete this by tomorrow.")
        print(f"✅ Urgent content: urgency={result.urgency:.2f}, deadline={result.deadline_proximity:.2f}")
        
        # Test normal content
        result = extractor.extract("Let's discuss this in the next meeting.")
        print(f"✅ Normal content: urgency={result.urgency:.2f}, deadline={result.deadline_proximity:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Temporal extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotional_extraction():
    """Test emotional dimension extraction."""
    print("🧪 Testing emotional extraction...")
    try:
        from src.extraction.dimensions.emotional_extractor import EmotionalDimensionExtractor
        
        extractor = EmotionalDimensionExtractor()
        
        # Test positive content
        result = extractor.extract("I'm really excited about this project! It's going to be amazing.")
        print(f"✅ Positive content: polarity={result.polarity:.2f}, intensity={result.intensity:.2f}")
        
        # Test negative content  
        result = extractor.extract("I'm disappointed with the results. This is not working.")
        print(f"✅ Negative content: polarity={result.polarity:.2f}, intensity={result.intensity:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Emotional extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dimension_analyzer():
    """Test the full dimension analyzer."""
    print("🧪 Testing dimension analyzer...")
    try:
        from src.extraction.dimensions.dimension_analyzer import get_dimension_analyzer
        
        analyzer = get_dimension_analyzer()
        
        # Test analysis
        import asyncio
        
        async def run_test():
            result = await analyzer.analyze("This is urgent! We need to complete this ASAP. I'm very excited about the results!")
            print(f"✅ Full analysis: {len(result.to_array())}D vector")
            dims = result.to_dict()
            print(f"   - Urgency: {dims['urgency']:.2f}")
            print(f"   - Polarity: {dims['polarity']:.2f}")
            print(f"   - Intensity: {dims['intensity']:.2f}")
            return True
            
        return asyncio.run(run_test())
        
    except Exception as e:
        print(f"❌ Dimension analyzer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_connection():
    """Test database connection."""
    print("🧪 Testing database connection...")
    try:
        from src.storage.sqlite.connection import DatabaseConnection
        
        # Try to create in-memory database
        db = DatabaseConnection(":memory:")
        print("✅ Database connection works")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("🚀 Testing Basic Pipeline Components")
    print("=" * 50)
    
    tests = [
        ("Models", test_models),
        ("Temporal Extraction", test_temporal_extraction), 
        ("Emotional Extraction", test_emotional_extraction),
        ("Dimension Analyzer", test_dimension_analyzer),
        ("Database Connection", test_database_connection),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 30)
        success = test_func()
        results.append((name, success))
        
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {name}")
        if success:
            passed += 1
            
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All basic components are working!")
        print("✅ Days 1-3 core functionality is complete")
        print("✅ Ready for storage layer and pipeline integration")
    else:
        print(f"\n⚠️  {len(results) - passed} component(s) need attention")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)