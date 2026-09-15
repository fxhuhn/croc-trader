from app.routes.security import _is_ip_whitelisted


def test_is_ip_whitelisted_exact_match() -> None:
    whitelist = ["127.0.0.1", "192.168.1.1"]
    assert _is_ip_whitelisted("127.0.0.1", whitelist) is True
    assert _is_ip_whitelisted("192.168.1.1", whitelist) is True
    assert _is_ip_whitelisted("10.0.0.1", whitelist) is False


def test_is_ip_whitelisted_wildcard_x() -> None:
    whitelist = ["172.16.x.x", "10.x.x.x"]
    assert _is_ip_whitelisted("172.16.0.1", whitelist) is True
    assert _is_ip_whitelisted("172.16.255.255", whitelist) is True
    assert _is_ip_whitelisted("172.17.0.1", whitelist) is False
    assert _is_ip_whitelisted("10.0.0.1", whitelist) is True
    assert _is_ip_whitelisted("11.0.0.1", whitelist) is False


def test_is_ip_whitelisted_wildcard_star() -> None:
    whitelist = ["192.168.*.*", "8.*.8.8"]
    assert _is_ip_whitelisted("192.168.1.1", whitelist) is True
    assert _is_ip_whitelisted("192.168.100.200", whitelist) is True
    assert _is_ip_whitelisted("192.167.1.1", whitelist) is False
    assert _is_ip_whitelisted("8.8.8.8", whitelist) is True
    assert _is_ip_whitelisted("8.4.8.8", whitelist) is True
    assert _is_ip_whitelisted("9.8.8.8", whitelist) is False


def test_is_ip_whitelisted_mixed_wildcards() -> None:
    whitelist = ["172.16.x.*"]
    assert _is_ip_whitelisted("172.16.0.1", whitelist) is True
    assert _is_ip_whitelisted("172.16.255.255", whitelist) is True
    assert _is_ip_whitelisted("172.17.0.1", whitelist) is False


def test_is_ip_whitelisted_empty_whitelist() -> None:
    assert _is_ip_whitelisted("127.0.0.1", []) is False


def test_is_ip_whitelisted_case_insensitive_x() -> None:
    whitelist = ["172.16.X.X"]
    assert _is_ip_whitelisted("172.16.0.1", whitelist) is True


def test_is_ip_whitelisted_invalid_segments() -> None:
    # If the whitelist pattern has more or fewer segments, it shouldn't crash
    whitelist = ["172.16.x", "172.16.x.x.x"]
    assert _is_ip_whitelisted("172.16.0.1", whitelist) is False


def test_is_ip_whitelisted_cidr() -> None:
    whitelist = ["192.168.1.0/24", "10.0.0.0/8"]
    assert _is_ip_whitelisted("192.168.1.1", whitelist) is True
    assert _is_ip_whitelisted("192.168.1.254", whitelist) is True
    assert _is_ip_whitelisted("192.168.2.1", whitelist) is False
    assert _is_ip_whitelisted("10.254.1.1", whitelist) is True
    assert _is_ip_whitelisted("11.0.0.1", whitelist) is False


def test_is_ip_whitelisted_invalid_cidr() -> None:
    whitelist = ["invalid/cidr/pattern"]
    assert _is_ip_whitelisted("192.168.1.1", whitelist) is False
