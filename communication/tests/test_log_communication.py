from communication.log_communication import log_communication
import pytest


def test_invalid_values():
    # invalid federated round
    with pytest.raises(ValueError):
        log_communication(federated_round=-1, sender=1, receiver=1, data=[], log_location=None)
    # invalid sender
    with pytest.raises(ValueError):
        log_communication(federated_round=1, sender=-2, receiver=1, data=[], log_location=None)
    # invalid receiver
    with pytest.raises(ValueError):
        log_communication(federated_round=1, sender=1, receiver=-2, data=[], log_location=None)
