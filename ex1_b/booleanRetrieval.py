from invertedIndex import InvertedIndex, PostingList


def boolean_and(posting_list1: PostingList, posting_list2: PostingList) -> PostingList:
    """Perform a Boolean AND of two PostingLists.
    This function assumes the posting lists are sorted by the 'internal_id' of documents
    and performs the intersection of the two lists by comparing the 'internal_id' values

    :param posting_list1: the first PostingList to be intersected.
    :param posting_list2: the second PostingList to be intersected.
    :return: A PostingList containing documents that are present in both posting_list1 and posting_list2.
    """
    i_pl1 = i_pl2 = 0
    len_pl1, len_pl2 = len(posting_list1), len(posting_list2)
    result: PostingList = []

    while i_pl1 < len_pl1 and i_pl2 < len_pl2:
        doc1 = posting_list1[i_pl1]
        doc2 = posting_list2[i_pl2]
        if doc1.internal_id == doc2.internal_id:  # Document exists in both posting lists
            result.append(posting_list1[i_pl1])
            i_pl1 += 1
            i_pl2 += 1

        elif doc1.internal_id < doc2.internal_id:  # Advance the index for the list with the smaller internal_id
            i_pl1 += 1

        else:
            i_pl2 += 1

    return result

def boolean_or(posting_list1: PostingList, posting_list2: PostingList) -> PostingList:
    """Perform a Boolean OR of two PostingLists.
    This function assumes the posting lists are sorted by the 'internal_id' of documents
    and performs the union of the two lists, ensuring no duplicate documents in the result.

    :param posting_list1: the first PostingList to be merged.
    :param posting_list2: the second PostingList to be merged.
    :return: A PostingList containing all unique documents from both posting_list1 and posting_list2.
    """
    i_pl1 = i_pl2 = 0
    len_pl1, len_pl2 = len(posting_list1), len(posting_list2)
    result: PostingList = []

    while i_pl1 < len_pl1 and i_pl2 < len_pl2:
        doc1 = posting_list1[i_pl1]
        doc2 = posting_list2[i_pl2]
        if doc1.internal_id == doc2.internal_id: # Document exists in both lists - add it once
            result.append(doc1)
            i_pl1 += 1
            i_pl2 += 1
        elif doc1.internal_id < doc2.internal_id: # Add the document from the first list and advance its index
            result.append(doc1)
            i_pl1 += 1
        else: # Add the document from the second list and advance its index
            result.append(doc2)
            i_pl2 += 1

    # Append any remaining documents in either list
    while i_pl1 < len_pl1:
        result.append(posting_list1[i_pl1])
        i_pl1 += 1

    while i_pl2 < len_pl2:
        result.append(posting_list2[i_pl2])
        i_pl2 += 1

    return result

def boolean_not(posting_list: PostingList, all_documents: PostingList) -> PostingList:
    """Perform a Boolean NOT operation on a given posting list.

    :param posting_list: The PostingList to exclude.
    :param all_documents: The complete list of all documents in the corpus.
    :return: A PostingList containing all documents not in the given posting list.
    """
    result: PostingList = []
    i_pl = 0
    len_pl = len(posting_list)

    for doc in all_documents:
        if i_pl < len_pl and doc.internal_id == posting_list[i_pl].internal_id:
            i_pl += 1  # Word appears in the document - skip
        else:
            result.append(doc)

    return result

def BooleanRetrieval(inverted_index: InvertedIndex, boolean_query: str) -> PostingList:
    """
    Process a Boolean query using the inverted index and return the matching PostingList.

    :param inverted_index: An instance of InvertedIndex containing the document mappings.
    :param boolean_query: A Boolean query string containing words and operators (AND, OR, NOT).
    :return: A PostingList of documents matching the Boolean query.
    """
    query_words = boolean_query.strip().split()
    stack = []

    for word in query_words:
        if word not in ["AND", "OR", "NOT"]: # word is an operand
            stack.append(inverted_index[word])
        else: # word is an operator
            if word == "AND":
                pl1 = stack.pop()
                pl2 = stack.pop()
                stack.append(boolean_and(pl1, pl2))
            elif word == "OR":
                pl1 = stack.pop()
                pl2 = stack.pop()
                stack.append(boolean_or(pl1, pl2))
            else: # word == "NOT"
                pl1 = stack.pop()
                not_result = boolean_not(pl1, inverted_index.all_docs)
                if len(stack) > 0: # Treat operator as AND NOT
                    pl2 = stack.pop()
                    stack.append(boolean_and(pl2, not_result))
                else: # Treat operator as NOT
                    stack.append(not_result)

    if len(stack) != 1:
        print(f"Invalid boolean query: {boolean_query}")

    return stack[0]


def execute_and_write_to_file():
    """Create an inverted index, read the BooleanQueries file and run BooleanRetrieval on each Boolean query.
    Write the results to a file"""
    import os
    
    # Find data directory - check multiple possible locations
    data_dir = None
    possible_data_dirs = ["data", "../ex1/data", "ex1/data", "../data"]
    for d in possible_data_dirs:
        if os.path.exists(d):
            data_dir = d
            break
    
    if data_dir is None:
        raise FileNotFoundError("Could not find data directory. Tried: " + ", ".join(possible_data_dirs))
    
    print(f"Using data directory: {data_dir}")
    index = InvertedIndex(data_dir)

    results = []
    # Find query file - check queries folder
    query_file = None
    possible_query_files = ["queries/query.txt", "queries/BooleanQueries.txt", 
                           "../ex1/queries/query.txt", "BooleanQueries.txt", "query.txt"]
    for qf in possible_query_files:
        if os.path.exists(qf):
            query_file = qf
            break
    
    if query_file is None:
        raise FileNotFoundError("Could not find query file. Tried: " + ", ".join(possible_query_files))
    
    print(f"Using query file: {query_file}")
    with open(query_file, "r") as file:
        queries = file.readlines()
    for query in queries:
        query = query.strip()
        if not query:
            results.append("")
            continue
        result = BooleanRetrieval(index, query)
        external_ids = " ".join(doc.external_id for doc in result)
        results.append(f"{external_ids}")

    output_file = "Part_2.txt"
    with open(output_file, "w") as file:
        content = '\n'.join(results)
        file.write(f"{content}")
    print(f"Results written to {output_file}")


if __name__ == '__main__':
    execute_and_write_to_file()


