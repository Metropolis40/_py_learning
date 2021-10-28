

def validate_user(username, minlen) :
    assert type(username) == str, "username must be a string"# 我们可以用assert 来确保正确的输入，避免code misbehave
    if minlen < 1 :
        raise ValueError("minlen must be at least 1") #here we raise a value error 
    if len(username) < minlen :
        return False
    if not username.isalnum() : # .isalnum means, is alphabatic or numeric
        return False
    return True

