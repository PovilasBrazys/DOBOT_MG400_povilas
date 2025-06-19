import string

def is_clean_api_key(api_key):
    allowed_chars = string.ascii_letters + string.digits
    for char in api_key:
        if char not in allowed_chars:
            print(f"Suspicious character found: '{char}' (Unicode: {ord(char)})")
            return False
    return True

# Example usage
key = input("Enter your API key: ")
if is_clean_api_key(key):
    print("API key looks clean!")
else:
    print("API key contains suspicious characters.")
