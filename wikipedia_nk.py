import wikipedia 

def search_wiki(query):
    global return_item
    try:
        result = str(wikipedia.summary(query, sentences=3, auto_suggest=False, redirect=True))
        return_item = result
        return return_item
    except wikipedia.exceptions.PageError as e:
        return_item = "Sorry sir, there appears to be no information on this topic"
        return return_item
    except wikipedia.exceptions.DisambiguationError as e:
        return_item = "I need a more specific query sir"
        return return_item