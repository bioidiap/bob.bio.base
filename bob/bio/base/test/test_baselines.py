import tempfile
import shutil
from click.testing import CliRunner
from bob.bio.base.script.baseline import baseline

def test_baselines():

    try:
        tmp_dir = tempfile.mkdtemp(prefix="bobtest_")
        runner = CliRunner()
        result = runner.invoke(baseline, args=('dummy', 'dummy', '-T', tmp_dir, '-R', tmp_dir))
        assertion_error_message = (
              'Command exited with this output and exception: `{}\' \n `{}\' \n'
              'If the output is empty, you can run this script locally to see '
              'what is wrong:\n'
              'bin/bob bio baseline dummy dummy -T /tmp/baseline -R /tmp/baseline'
              ''.format(result.output, result.exception))
        assert result.exit_code == 0, assertion_error_message

    finally:
        shutil.rmtree(tmp_dir)
