import tempfile
import os
import shutil
from click.testing import CliRunner
from bob.bio.base.script.annotate import annotate
from bob.bio.base.annotator import Annotator, FailSafe
from bob.db.base import read_annotation_file
from bob.pipelines import Sample


def test_annotate():

    try:
        tmp_dir = tempfile.mkdtemp(prefix="bobtest_")
        runner = CliRunner()
        result = runner.invoke(annotate, args=(
            '-d', 'dummy', '-a', 'dummy', '-g', 'dev', '-o', tmp_dir))
        assertion_error_message = (
            'Command exited with this output: `{}\' \n'
            'If the output is empty, you can run this script locally to see '
            'what is wrong:\n'
            'bin/bob bio annotate -vvv --force -d dummy -a dummy -o /tmp/temp_annotations'
            ''.format(result.output))
        assert result.exit_code == 0, assertion_error_message

        # test if annotations exist
        for dirpath, dirnames, filenames in os.walk(tmp_dir):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                annot = read_annotation_file(path, 'json')
                assert annot['topleft'] == [0, 0]
                # size of atnt images
                assert annot['bottomright'] == [112, 92]
    finally:
        shutil.rmtree(tmp_dir)


class dummy_extra_key_annotator(Annotator):
    def transform(self, samples, **kwargs):
        for s in samples:
            s.annotations = {'leye': 0, 'reye': 0, 'topleft': 0}
        return samples


def test_failsafe():
    annotator = FailSafe([dummy_extra_key_annotator()], ['leye', 'reye'])
    samples = [Sample(data=1, key="dummy_sample")]
    annotated = annotator(samples)[0]
    assert all(key in annotated.annotations for key in ['leye', 'reye', 'topleft'])

    annotator = FailSafe([dummy_extra_key_annotator()], ['leye', 'reye'], True)
    samples = [Sample(data=1, key="dummy_sample")]
    annotated = annotator(samples)[0]
    assert all(key in annotated.annotations for key in ['leye', 'reye'])
    samples = [Sample(data=1, key="dummy_sample")]
    annotated = annotator(samples)[0]
    assert all(key not in annotated.annotations for key in ['topleft'])
