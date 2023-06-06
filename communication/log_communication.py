
def log_communication(federated_round : int, sender : int, receiver: int, data : list, log_location: str):
    """
    Log the communication to a csv.

    Parameters
    ----------
    data : list
        _description_
    federated_round : int
        The ID of the federated round.
    sender : int
        The ID of the sender of the data. 
        The server has the ID = -1 and the clients have IDs >= 0.
    receiver : int
        The ID of the receiver of the data. 
        The server has the ID = -1 and the clients have IDs >= 0.

    Raises
    ------
    ValueError
        Raise if the federated round, sender or receiver have invalid IDs.
    """    
    if sender < -1 or receiver < -1:
        raise ValueError("The ID of the receiver or sender is smaller than -1 and thus invalid.")
    if federated_round < 0:
        raise ValueError("The ID of federated round is negative which is invalid.")
    

    amount_of_bytes = 42
    _save_row_to_csv([federated_round, sender, receiver, amount_of_bytes], path=log_location)
    raise NotImplementedError("Computation of amount_of_bytes is not implemented yet.")
    

def _save_row_to_csv(row : list, path : str):
    """
    Save row seperated by commas to the specified path.

    Parameters
    ----------
    row : list
        The row that will be saved.
    path : str
        The location of the csv.

    Raises
    ------
    ValueError
        Raise if path is None.
    """    
    if path is not None:
        with open("test.txt", "a") as f:
            f.write(",".join([str(c) for c in row]))
    else:
        raise ValueError("No path provided and thus communication will not be logged.")