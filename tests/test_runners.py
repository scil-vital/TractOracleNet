def test_ttl_track(script_runner):
    # Call 'ttl_track.py' from the command line and assert that it
    # runs without errors

    ret = script_runner.run('predictor.py', '--help')
    assert ret.success
