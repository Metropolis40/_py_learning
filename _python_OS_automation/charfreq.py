def character_frequency(filename) :
    """counts the freq of each char in the given file."""
    # first try to open the file
    try :
        f = open(filename)
    except OSError :
        return None


    # now process the file 
    characters = {}
    for line in f :
        for char in line :
            characters[char] = characters.get(char, 0) +1 
    f.close()

    return characters
    