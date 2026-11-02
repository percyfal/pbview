import panel as pn
import pytest

import pbview.app as app_mod
from pbview.model.datastore import DataStore


def test_serve_builds_one_view_per_session(
    make_datastore_path, sample_sets, monkeypatch
):
    path = make_datastore_path(
        with_sum=True, with_missing=True, sample_sets=sample_sets
    )
    served = {}
    monkeypatch.setattr(app_mod.pn, "serve", lambda obj, **kw: served.update(obj=obj))
    opened, views = [], []
    real_store, real_view = DataStore, app_mod.DataStoreView
    monkeypatch.setattr(
        app_mod, "DataStore", lambda *a, **k: opened.append(1) or real_store(*a, **k)
    )

    def make_view(*args, **kwargs):
        views.append(real_view(*args, **kwargs))
        return views[-1]

    monkeypatch.setattr(app_mod, "DataStoreView", make_view)

    app_mod.serve(servable=False, path=str(path), max_positions=0, port=1)

    # a function that Panel calls for each session, not a view built once
    create = served["obj"]
    assert callable(create)
    assert not isinstance(create, pn.viewable.Viewable)
    first, second = create(), create()
    assert first is not second
    assert len(opened) == 1  # one data store, shared
    assert len(views) == 2
    # the state and the Bokeh figure belong to one session each
    a, b = views
    assert a.state is not b.state
    assert a.pages["Summary"]._figure is not b.pages["Summary"]._figure
    before = b.state.coord
    a.state.contigs = ["chr1"]
    assert b.state.coord == before


@pytest.mark.parametrize(
    ("address", "loopback"),
    [
        (None, True),
        ("localhost", True),
        ("127.0.0.1", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("10.1.2.3", False),
        ("node123.cluster", False),
    ],
)
def test_is_loopback(address, loopback):
    assert app_mod.is_loopback(address) is loopback


@pytest.fixture
def served(make_datastore_path, sample_sets, monkeypatch):
    """Serve a store with `pn.serve` replaced; the arguments are collected."""
    path = make_datastore_path(
        with_sum=True, with_missing=True, sample_sets=sample_sets
    )
    collected = {}
    monkeypatch.setattr(
        app_mod.pn, "serve", lambda obj, **kw: collected.update(obj=obj, **kw)
    )

    def serve(**kw):
        collected.clear()
        app_mod.serve(servable=False, path=str(path), max_positions=0, **kw)
        return collected

    return serve


def test_serve_passes_the_address_and_the_origins_on(served):
    kw = served(port=8080, address="0.0.0.0", websocket_origin=["localhost:9000"])
    assert kw["address"] == "0.0.0.0"
    assert kw["port"] == 8080
    # the origins asked for, and the one Panel allows by default
    assert kw["websocket_origin"] == ["localhost:9000", "localhost:8080"]


def test_serve_keeps_the_default_origin_once(served):
    kw = served(port=8080, websocket_origin=["localhost:8080", "localhost:9000"])
    assert kw["websocket_origin"] == ["localhost:8080", "localhost:9000"]


@pytest.mark.parametrize("address", [None, ""])
def test_serve_listens_on_localhost_by_default(served, address):
    # Panel's own default is all interfaces; the app has no login
    kw = served(port=8080, address=address)
    assert kw["address"] == "localhost"
    assert kw.get("websocket_origin") is None  # Panel's default: localhost:port


@pytest.mark.parametrize("address", ["0.0.0.0", "10.1.2.3", "node123.cluster"])
def test_serve_warns_that_there_is_no_login(served, caplog, address):
    with caplog.at_level("WARNING"):
        served(port=8080, address=address)
    assert any(
        "no login" in r.getMessage() and address in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize("address", [None, "localhost", "127.0.0.1"])
def test_serve_is_silent_on_localhost(served, caplog, address):
    with caplog.at_level("WARNING"):
        served(port=8080, address=address)
    assert not [r for r in caplog.records if "no login" in r.getMessage()]


def test_serve_says_where_it_serves(served, caplog):
    with caplog.at_level("INFO"):
        served(port=8080, address="0.0.0.0")
    assert "Serving at http://0.0.0.0:8080" in caplog.text
    caplog.clear()
    with caplog.at_level("INFO"):
        served(port=8081)
    assert "Serving at http://localhost:8081" in caplog.text
