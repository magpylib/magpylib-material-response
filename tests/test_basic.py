import magpylib_response


def test_version():
    assert isinstance(magpylib_response.__version__, str)
