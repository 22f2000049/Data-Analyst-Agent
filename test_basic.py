try:
    import flask
    print("✅ Flask installed")
except ImportError:
    print("❌ Flask missing")

try:
    import pandas
    print("✅ Pandas installed") 
except ImportError:
    print("❌ Pandas missing")

try:
    import matplotlib
    print("✅ Matplotlib installed")
except ImportError:
    print("❌ Matplotlib missing")

try:
    import requests
    print("✅ Requests installed")
except ImportError:
    print("❌ Requests missing")

try:
    import google.generativeai
    print("✅ Gemini AI installed")
except ImportError:
    print("❌ Gemini AI missing")

print("\n🚀 Ready to start the application!")