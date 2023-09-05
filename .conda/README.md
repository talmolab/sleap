This folder defines the conda package build for Linux and Windows. There are runners for both Linux and Windows on GitHub Actions, but it is faster to experiment with builds locally first.

To build, first go to the base repo directory and install the build environment:

```
mamba env create -f environment_build.yml -n sleap_build && mamba activate sleap_build
```

And finally, run the build command pointing to this directory:

```
conda build .conda --output-folder build -c conda-forge -c nvidia -c https://conda.anaconda.org/sleap/ -c anaconda
```

To install the local package:

```
mamba create -n sleap_0 -c conda-forge -c nvidia -c ./build -c https://conda.anaconda.org/sleap/ -c anaconda sleap=x.x.x
```

replacing x.x.x with the version of SLEAP that you just built.

> Note(LM) 09/05/2023: I've run the following merge error
>```
>(sleap_ci) talmolab@talmolab-01-ubuntu:~/sleap-estimates-animal-poses/pull-requests/sleap$ conda build .conda --output-folder build
>WARNING: No numpy version specified in conda_build_config.yaml.  Falling back to default numpy value of 1.22
>Copying /home/talmolab/sleap-estimates-animal-poses/pull-requests/sleap to /home/talmolab/micromamba/envs/sleap_ci/conda-bld/sleap_1693933608490/work/
>Adding in variants from internal_defaults
>Attempting to finalize metadata for sleap
>Collecting package metadata (repodata.json): ...working... done
>Solving environment: ...working... failed
>
># >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<
>
>    Traceback (most recent call last):
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/exception_handler.py", line 17, in __call__
>        return func(*args, **kwargs)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/cli/main.py", line 64, in main_subshell
>        exit_code = do_call(args, parser)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/cli/conda_argparse.py", line 143, in do_call
>        result = plugin_subcommand.action(getattr(args, "_args", args))
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/plugin.py", line 10, in build
>        execute(*args, **kwargs)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/cli/main_build.py", line 568, in execute
>        outputs = api.build(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/api.py", line 253, in build
>        return build_tree(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/build.py", line 3804, in build_tree
>        packages_from_this = build(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/build.py", line 2474, in build
>        output_metas = expand_outputs([(m, need_source_download, need_reparse_in_env)])
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/render.py", line 923, in expand_outputs
>        for output_dict, m in deepcopy(_m).get_output_metadata_set(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/metadata.py", line 2574, in get_output_metadata_set
>        conda_packages = finalize_outputs_pass(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/metadata.py", line 934, in finalize_outputs_pass
>        fm = finalize_metadata(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/render.py", line 637, in finalize_metadata
>        build_unsat, host_unsat = add_upstream_pins(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/render.py", line 478, in add_upstream_pins
>        host_deps, host_unsat, extra_run_specs_from_host = _read_upstream_pin_files(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/render.py", line 431, in _read_upstream_pin_files
>        deps, actions, unsat = get_env_dependencies(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/render.py", line 151, in get_env_dependencies
>        actions = environ.get_install_actions(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda_build/environ.py", line 900, in get_install_actions
>        actions = install_actions(prefix, index, specs, force=True)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/common/io.py", line 84, in decorated
>        return f(*args, **kwds)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/plan.py", line 535, in install_actions
>        txn = solver.solve_for_transaction(prune=prune, ignore_pinned=not pinned)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/core/solve.py", line 154, in solve_for_transaction
>        unlink_precs, link_precs = self.solve_for_diff(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/core/solve.py", line 215, in solve_for_diff
>        final_precs = self.solve_final_state(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/core/solve.py", line 380, in solve_final_state
>        ssc = self._add_specs(ssc)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/core/solve.py", line 874, in _add_specs
>        conflicts = ssc.r.get_conflicting_specs(
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/resolve.py", line 1270, in get_conflicting_specs
>        C = r2.gen_clauses()
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/common/io.py", line 84, in decorated
>        return f(*args, **kwds)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/resolve.py", line 1055, in gen_clauses
>        for ms in self.ms_depends(prec):
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/resolve.py", line 929, in ms_depends
>        deps = [MatchSpec(d) for d in prec.combined_depends]
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/models/records.py", line 366, in combined_depends
>        result = {ms.name: ms for ms in MatchSpec.merge(self.depends)}
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/models/match_spec.py", line 493, in merge
>        reduce(lambda x, y: x._merge(y, union), group)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/models/match_spec.py", line 493, in <lambda>
>        reduce(lambda x, y: x._merge(y, union), group)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/models/match_spec.py", line 525, in _merge
>        final = this_component.merge(that_component)
>      File "/home/talmolab/micromamba/envs/sleap_ci/lib/python3.10/site-packages/conda/models/match_spec.py", line 815, in merge
>        raise ValueError(
>    ValueError: Incompatible component merge:
>      - '*mpich*'
>      - 'mpi_mpich_*'
>```
>decribed in detail here https://github.com/conda/conda/issues/11442 and needed to apply the changes from conda/conda#11612
>https://github.com/conda/conda/blob/d859312460760db6c6a9361500f617e308e38f07/conda/models/match_spec.py#L900-L964
>to get the build working.