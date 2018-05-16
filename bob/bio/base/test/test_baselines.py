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
              'Command exited with this output: `{}\' \n'
              'If the output is empty, you can run this script locally to see '
              'what is wrong:\n'
              'bin/bob bio baseline  -d dummy -a dummy -o /tmp/temp_annotations'
              ''.format(result.output))
        assert result.exit_code == 0, assertion_error_message

    finally:
        shutil.rmtree(tmp_dir)
