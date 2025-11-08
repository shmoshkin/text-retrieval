import os


class BooleanRetrieval:
    def __init__(self, r_index, queries_path):
        self.r_index = r_index
        self._queries = self.load_queries(queries_path)

    def load_queries(self, queries_path):
        if not os.path.isfile(queries_path):
            raise ValueError(f"Queries file {queries_path} does not exist")

        q_list = []
        # return a list of lines
        with open(queries_path, "r") as f:
            for line in f:
                q_list.append(line.strip())

        return q_list

    # Generator for the queries
    def queries(self):
        for query in self._queries:
            yield query

    # Function to process a single RPN query
    def process_query_rpn(self, query):
        """
        Processes a query in Reverse Polish Notation and calls the appropriate functions.
        """
        stack = []
        operators = {"AND": self.op_and,
                     "OR": self.op_or,
                     "NOT": self.op_and_not
                     }

        if len(query.split()) == 1:
            term = query.split()[0]
            if term not in self.r_index.indices:
                return []  # Return an empty result if the term is not in the index
            return self.op_simple(term)

        for token in query.split():
            if token in operators:  # If token is an operator
                if token == "NOT":
                    term2 = stack.pop()  # NOT is treated as AND NOT
                    term2 = [] if term2 not in self.r_index.indices else self.r_index.indices[term2]
                    term1 = stack.pop()

                    if isinstance(term1, list):
                        result = operators[token](term1, term2)
                    else:
                        term1 = [] if term1 not in self.r_index.indices else self.r_index.indices[term1]
                        result = operators[token](term1, term2)
                else:
                    term2 = stack.pop()
                    term2 = [] if term2 not in self.r_index.indices else self.r_index.indices[term2]
                    term1 = stack.pop()

                    if isinstance(term1, list):
                        short_list, long_list = self._return_short_long_list(
                            term1, term2)
                    else:
                        term1 = [] if term1 not in self.r_index.indices else self.r_index.indices[term1]
                        short_list, long_list = self._return_short_long_list(
                            term1, term2)

                    result = operators[token](short_list, long_list)
                # Push the result back onto the stack
                stack.append(result)
            else:
                # If token is a term, push it onto the stack
                stack.append(token)

        # The stack should contain the final result
        return stack.pop() if stack else None

    def op_and(self, s_list, l_list):
        """AND operation between two sorted lists of tuples [(internal_id, external_id)...]."""
        result = []
        i, j = 0, 0  # Pointers for s_list and l_list

        while i < len(s_list) and j < len(l_list):
            if s_list[i][0] == l_list[j][0]:
                # Internal IDs match, add the external ID to the result
                if not result or result[-1][0] != s_list[i][0]:
                    result.append(s_list[i])
                i += 1
                j += 1
            elif s_list[i][0] < l_list[j][0]:
                # Move the pointer for s_list forward
                i += 1
            else:
                # Move the pointer for l_list forward
                j += 1

        return result
    
    def op_or(self, s_list, l_list):
        """OR operation between two lists of tuples [(internal_id, external_id)...]. Ensures no duplicates."""
        result = []
        i, j = 0, 0  # Pointers for s_list and l_list

        def add_to_result(result, item):
            """Helper function to add an item to the result list if it's not a duplicate."""
            if not result or result[-1][0] != item[0]:
                result.append(item)

        while i < len(s_list) and j < len(l_list):
            if s_list[i][0] == l_list[j][0]:
                add_to_result(result, s_list[i])
                i += 1
                j += 1
            elif s_list[i][0] < l_list[j][0]:
                add_to_result(result, s_list[i])
                i += 1
            else:
                add_to_result(result, l_list[j])
                j += 1

        # Append any remaining elements from s_list, ensuring no duplicates
        while i < len(s_list):
            add_to_result(result, s_list[i])
            i += 1

        # Append any remaining elements from l_list, ensuring no duplicates
        while j < len(l_list):
            add_to_result(result, l_list[j])
            j += 1

        return result


    def op_and_not(self, yes_list, not_list):
        """AND NOT operation between two lists of tuples [(internal_id, external_id)...]."""
        result = []
        i, j = 0, 0

        while i < len(yes_list) and j < len(not_list):
            if yes_list[i][0] == not_list[j][0]:
                # Skip all matching elements in yes_list
                while i < len(yes_list) and yes_list[i][0] == not_list[j][0]:
                    i += 1
                j += 1
            elif yes_list[i][0] < not_list[j][0]:
                result.append(yes_list[i])
                i += 1
            else:
                j += 1

        # Add remaining elements from yes_list to result
        while i < len(yes_list):
            result.append(yes_list[i])
            i += 1

        return result


    def op_simple(self, term):
        return [doc for doc in self.r_index.indices[term]]

    @staticmethod
    def _return_short_long_list(list1, list2):
        return (list1, list2) if len(list1) < len(list2) else (list2, list1)
