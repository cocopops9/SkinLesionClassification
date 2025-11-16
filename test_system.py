#!/usr/bin/env python
"""
Test script to verify all components of the Enhanced Melanoma Detection System.
Run this before starting the main application to ensure everything is configured correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    modules = [
        ('streamlit', 'Streamlit'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('bcrypt', 'bcrypt'),
        ('matplotlib', 'Matplotlib')
    ]
    
    failed = []
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError as e:
            print(f"  ❌ {display_name}: {e}")
            failed.append(display_name)
    
    return len(failed) == 0

def test_custom_modules():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    
    modules = [
        'database',
        'classification', 
        'explainability',
        'image_validator'
    ]
    
    failed = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}.py")
        except ImportError as e:
            print(f"  ❌ {module_name}.py: {e}")
            failed.append(module_name)
    
    return len(failed) == 0

def test_database():
    """Test database functionality."""
    print("\nTesting database...")
    
    try:
        from database import DatabaseManager, User
        
        # Initialize database
        DatabaseManager.init_db()
        print("  ✅ Database initialized")
        
        # Test user creation
        import uuid
        test_username = f"test_user_{uuid.uuid4().hex[:8]}"
        test_user = DatabaseManager.create_user(
            username=test_username,
            password="TestPassword123",
            email=f"{test_username}@test.com"
        )
        
        if test_user:
            print(f"  ✅ User creation successful (ID: {test_user.id[:8]}...)")
            
            # Test authentication
            auth_user = DatabaseManager.authenticate_user(test_username, "TestPassword123")
            if auth_user:
                print("  ✅ Authentication successful")
            else:
                print("  ❌ Authentication failed")
                return False
        else:
            print("  ❌ User creation failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False
    
    return True

def test_tensorflow():
    """Test TensorFlow configuration."""
    print("\nTesting TensorFlow...")
    
    try:
        import tensorflow as tf
        
        print(f"  ✅ TensorFlow version: {tf.__version__}")
        
        # Check if GPU is available (optional)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✅ GPU available: {len(gpus)} device(s)")
        else:
            print("  ℹ️ No GPU detected (CPU mode)")
        
        # Test basic operation
        test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = tf.reduce_sum(test_tensor)
        
        if result.numpy() == 10.0:
            print("  ✅ TensorFlow operations working")
        else:
            print("  ❌ TensorFlow operations failed")
            return False
            
    except Exception as e:
        print(f"  ❌ TensorFlow error: {e}")
        return False
    
    return True

def test_image_validation():
    """Test image validation functionality."""
    print("\nTesting image validation...")
    
    try:
        from image_validator import ImageValidator
        import numpy as np
        
        validator = ImageValidator()
        print("  ✅ Image validator initialized")
        
        # Create a test image (random noise)
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test validation
        results = validator.validate_image(test_img)
        
        if isinstance(results, dict) and 'is_valid' in results:
            print("  ✅ Image validation working")
        else:
            print("  ❌ Image validation failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Image validation error: {e}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist or can be created."""
    print("\nTesting directories...")
    
    directories = [
        'models',
        'user_uploads',
        'heatmaps'
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        try:
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ {dir_name}/")
        except Exception as e:
            print(f"  ❌ {dir_name}/: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Enhanced Melanoma Detection System - Component Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Custom Modules Test", test_custom_modules),
        ("Database Test", test_database),
        ("TensorFlow Test", test_tensorflow),
        ("Image Validation Test", test_image_validation),
        ("Directory Test", test_directories)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to run.")
        print("Run the application with: streamlit run app.py")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running the application.")
        print("Check the README.md for troubleshooting tips.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
