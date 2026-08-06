import hashlib
import pytest


from ntxter.adapters.data.anonimization import SecureHasherFromList


@pytest.fixture
def key_fixed():
        key_test = "key_for_testing"
        key = hashlib.sha256(key_test.encode('utf-8')).digest()
        return key


class TestSecureHasherFromList:
    @pytest.mark.parametrize(
            ("elements", "hash_expected"),
            (
                (
                    ["Hi"],
                    "cdd749bb9a2a1f50de270d49147e20b196fbc1632d0e4a31ab91610ab6dc8221"
                ),
                (
                    ["Hi", "bye"],
                    "4780aaf48bababc62edf51947774bb26401b5669eef1be51e64411bc3ef0db18"
                ),
                (
                    ["Hi", "hi"],
                    "e40d99e077bbfb3802485b77e272bde215a420760292e9fb2fb3dc960b2598c3"
                ),
                (
                    ["Hi", "hi", "bye", "ciao"],
                    "c295b3854793b86addb08df60fec9f7e3b0a16372560862a0897b9d0f441a81a"
                ),
                (
                    ["Hi", "hi", "bye", 3.1416],
                    "9bb7fe20ba703242715c15b9fd75596c1f9b57a9942ed4abe74e6deee644a76e"
                ),
            )                   
    )
    def test_hashing_from_list(self, key_fixed, elements, hash_expected):
        with SecureHasherFromList(key_fixed) as hasher:
            assert hasher(elements) == hash_expected

    @pytest.mark.parametrize(
        ("elements"),
        (
            "Hi",
            5,
            2.4
        )
    )
    def test_hashing_error_not_list(self, key_fixed, elements):
        with SecureHasherFromList(key_fixed) as hasher:
            with pytest.raises(ValueError):
                 hasher(elements)