import subprocess
import os
import pytest


class TestDocs:
    @pytest.mark.dependency(name='test_apidoc')
    def test_apidoc(self):
        os.environ["SPHINX_APIDOC_OPTIONS"] = "members,inherited-members"

        apidoc_out = subprocess.run(
            ["sphinx-apidoc", "-f", "--no-toc", "--templatedir=docs/source/_templates", "-e", "-o", "docs/source/api",
             "mlexp"], check=True, capture_output=True).stdout
        apidoc_out = str(apidoc_out)

        if ("warning" in apidoc_out) | ("error" in apidoc_out):
            raise Exception(
                "Errors or warnings were raised during running sphinx-apidoc command. Run docs/build_docs.sh to check for warnings/errors.")

    @pytest.mark.dependency(name='test_build', depends=['test_apidoc'])
    def test_build(self):
        build_out = subprocess.run(["sphinx-build", "-M", "html", "docs/source", "docs/build"], check=True,
                                   capture_output=True).stdout
        build_out = str(build_out)
        if ("warning" in build_out) | ("error" in build_out):
            raise Exception(
                "Errors or warnings were raised during running sphinx-build command. Run docs/build_docs.sh to check for warnings/errors.")

        if ("no targets are out of date" not in build_out):
            raise Exception("Documentation was not updated. Documentation is now updated, don't forget to commit it.")
