# test_step2.py
print("🧪 TESTING STEP 2: Data Loader")

try:
    from config import Config
    from utils.data_loader import IDSDataLoader
    
    config = Config()
    data_loader = IDSDataLoader(config)
    
    print("✓ DataLoader imported successfully!")
    
    # Test the data loading
    success = data_loader.test_data_loading()
    
    if success:
        print("\n🎉 STEP 2 COMPLETED SUCCESSFULLY!")
        print("You can now move to Step 3")
    else:
        print("\n❌ Please fix the data loading issues above")
        
except Exception as e:
    print(f"❌ STEP 2 FAILED: {e}")