"""
Tests for hpvsim/settings.py — Options class, style management, and configuration.
"""

import sciris as sc
import hpvsim as hpv
import pytest

hpv.options.set(interactive=False)


def test_options_get_set():
    """Test basic get/set on options."""
    # Get current value
    orig = hpv.options['verbose']

    # Set and verify
    hpv.options.set(verbose=0.1)
    assert hpv.options['verbose'] == 0.1

    # Restore
    hpv.options.set(verbose=orig)


def test_options_context():
    """Test options context manager."""
    orig_verbose = hpv.options['verbose']

    with hpv.options.context(verbose=0):
        assert hpv.options['verbose'] == 0

    # Should be restored after context
    assert hpv.options['verbose'] == orig_verbose


def test_options_callable():
    """Test calling options as a function."""
    orig = hpv.options['verbose']
    hpv.options(verbose=0)
    assert hpv.options['verbose'] == 0
    hpv.options(verbose=orig)


def test_options_to_dict():
    """Test converting options to dict."""
    d = hpv.options.to_dict()
    assert isinstance(d, dict)
    assert 'verbose' in d
    assert 'interactive' in d


def test_options_repr():
    """Test string representation."""
    r = repr(hpv.options)
    assert isinstance(r, str)
    assert len(r) > 0


def test_options_disp():
    """Test display method."""
    # Should not raise
    hpv.options.disp()


def test_options_get_default():
    """Test getting default values."""
    default = hpv.options.get_default('verbose')
    assert default is not None


def test_options_changed():
    """Test checking if option has changed from default."""
    hpv.options.set(verbose=hpv.options.get_default('verbose'))
    assert not hpv.options.changed('verbose')

    hpv.options.set(verbose=99.9)
    assert hpv.options.changed('verbose')

    # Restore
    hpv.options.set(verbose=hpv.options.get_default('verbose'))


def test_options_help():
    """Test options help output — prints to stdout, returns None."""
    # Should run without error
    hpv.options.help()
    hpv.options.help(detailed=True)


def test_options_save_load(tmp_path):
    """Test saving and loading options."""
    filepath = str(tmp_path / 'test_options.json')
    hpv.options.save(filepath)

    # Modify an option
    orig = hpv.options['verbose']
    hpv.options.set(verbose=99.9)

    # Load should restore
    hpv.options.load(filepath)
    assert hpv.options['verbose'] == orig


def test_options_style():
    """Test style handling."""
    # with_style should work as context manager
    with hpv.options.with_style(fontsize=14):
        pass  # Just verify it doesn't raise

    # use_style should work
    hpv.options.use_style(fontsize=12)


def test_options_set_matplotlib_global():
    """Test setting matplotlib globals."""
    hpv.options.set_matplotlib_global('dpi', 100)


def test_invalid_option():
    """Test that setting an invalid option raises an error."""
    with pytest.raises(sc.KeyNotFoundError):
        hpv.options.set(nonexistent_option=42)


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()
    test_options_get_set()
    test_options_context()
    test_options_callable()
    test_options_to_dict()
    test_options_repr()
    test_options_disp()
    test_options_get_default()
    test_options_changed()
    test_options_help()
    test_options_style()
    test_options_set_matplotlib_global()
    test_invalid_option()
    sc.toc(T)
    print('Done.')
