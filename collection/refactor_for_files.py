import argparse
import json
import logging
import os
import traceback

from dotenv import load_dotenv
from multiprocessing import Pool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()
NEED_FILES = [
    "src/BulletDynamics/ConstraintSolver/btSolverBody.h",
    "src/LinearMath/btVector3.h",
    "src/Bullet3Common/b3Matrix3x3.h",
    "src/Bullet3Common/b3QuadWord.h",
    "src/LinearMath/btVector3.h",
    "src/BulletCollision/BroadphaseCollision/btOverlappingPairCache.h",
    "src/Bullet3Common/b3AlignedObjectArray.h",
    "src/LinearMath/btQuaternion.h",
    "src/Bullet3Common/b3Matrix3x3.h",
    "src/LinearMath/btQuaternion.h",
    "src/LinearMath/btMatrix3x3.h",
    "src/LinearMath/btQuaternion.h",
    "src/LinearMath/btTransform.h",
    "src/LinearMath/btQuaternion.h",
    "src/LinearMath/btMatrix3x3.h",
    "src/LinearMath/btVector3.h",
    "src/LinearMath/btMatrix3x3.h",
    "src/LinearMath/btQuaternion.h",
    "src/LinearMath/btTransform.h",
    "src/LinearMath/btTransform.h",
    "src/LinearMath/btTransform.h",
    "src/LinearMath/btMatrix3x3.h",
    "src/Bullet3Common/b3Quaternion.h",
    "apps/cmp.c",
    "ssl/statem/statem_lib.c",
    "crypto/modes/ocb128.c",
    "crypto/x509/v3_addr.c",
    "crypto/x509/v3_addr.c",
    "engines/e_loader_attic.c",
    "apps/req.c",
    "ssl/quic/quic_channel.c",
    "ssl/quic/quic_engine.c",
    "ssl/t1_lib.c",
    "ssl/tls13_enc.c",
    "ssl/ssl_lib.c",
    "crypto/provider_child.c",
    "crypto/property/property.c",
    "crypto/evp/keymgmt_lib.c",
    "crypto/evp/keymgmt_lib.c",
    "crypto/rsa/rsa_sp800_56b_gen.c",
    "crypto/evp/evp_lib.c",
    "crypto/store/store_lib.c",
    "crypto/x509/x509_vfy.c",
    "crypto/x509/x509_vfy.c",
    "ssl/quic/quic_channel.c",
    "crypto/threads_pthread.c",
    "crypto/provider_conf.c",
    "ssl/quic/quic_txp.c",
    "crypto/evp/asymcipher.c",
    "ssl/t1_lib.c",
    "ssl/quic/quic_record_rx.c",
    "ssl/quic/quic_stream_map.c",
    "crypto/x509/v3_purp.c",
    "ssl/quic/quic_txpim.c",
    "crypto/ts/ts_rsp_sign.c",
    "crypto/stack/stack.c",
    "ssl/quic/quic_record_shared.c",
    "ssl/statem/statem.c",
    "crypto/x509/v3_sxnet.c",
    "crypto/rsa/rsa_lib.c",
    "ssl/quic/quic_engine.c",
    "ssl/quic/quic_types.c",
    "ssl/quic/quic_srtm.c",
    "crypto/property/property.c",
    "ssl/quic/quic_ackm.c",
    "ssl/quic/quic_rstream.c",
    "crypto/rsa/rsa_lib.c",
    "crypto/store/store_meth.c",
    "crypto/http/http_lib.c",
    "crypto/deterministic_nonce.c",
    "crypto/property/defn_cache.c",
    "ssl/quic/quic_record_tx.c",
    "ssl/quic/quic_lcidm.c",
    "crypto/objects/obj_dat.c",
    "ssl/quic/quic_fc.c",
    "ssl/quic/quic_txp.c",
    "ssl/quic/quic_channel.c",
    "ssl/ssl_local.h",
    "ssl/t1_lib.c",
    "ssl/quic/quic_demux.c",
    "ssl/quic/quic_sf_list.c",
    "ssl/statem/statem_srvr.c",
    "crypto/store/store_lib.c",
    "ssl/quic/quic_fc.c",
    "crypto/x509/x509_vpm.c",
    "src/cli_common.c",
    "src/dict.c",
    "src/rio.c",
    "src/server.c",
    "src/script_lua.c",
    "src/script_lua.c",
    "src/script_lua.c",
    "src/server.c",
    "src/script_lua.c",
    "src/script_lua.c",
    "src/eval.c",
    "src/function_lua.c",
    "src/threads_mngr.c",
    "src/evict.c",
    "src/redis-cli.c",
    "src/redis-cli.c",
    "src/redis-cli.c",
    "src/bio.c",
    "src/util.c",
    "src/connection.h",
    "src/networking.c",
    "src/cluster_legacy.c",
    "src/util.c",
    "src/t_zset.c",
    "src/t_zset.c",
    "src/networking.c",
    "src/kvstore.c",
    "src/util.c",
    "src/dict.c",
    "src/dict.c",
    "src/debug.c",
    "src/db.c",
    "src/blocked.c",
    "src/networking.c",
    "src/socket.c",
    "src/server.c",
    "src/monotonic.c",
    "src/functions.c",
    "src/networking.c",
    "src/zmalloc.c",
    "src/kvstore.c",
    "src/kvstore.c",
    "src/kvstore.c",
    "src/kvstore.c",
    "src/dict.c",
    "src/anet.c",
    "src/rax.c",
    "src/server.c",
    "src/acl.c",
    "src/db.c",
    "src/cluster_legacy.c",
    "src/object.c",
    "src/evict.c",
    "src/t_zset.c",
    "src/kvstore.c",
    "src/server.c",
    "src/db.c",
    "src/db.c",
    "src/kvstore.c",
    "src/kvstore.c",
    "src/db.c",
    "src/blocked.c",
    "src/slowlog.c",
    "src/server.c",
    "src/t_stream.c",
    "src/dict.c",
    "src/t_stream.c",
    "src/pubsub.c",
    "llvm/lib/ProfileData/SampleProfReader.cpp",
    "llvm/include/llvm/Demangle/Utility.h",
    "llvm/lib/ExecutionEngine/Orc/OrcABISupport.cpp",
    "llvm/lib/ProfileData/SampleProfReader.cpp",
    "llvm/lib/Transforms/IPO/FunctionImport.cpp",
    "llvm/lib/Support/ThreadPool.cpp",
    "llvm/lib/Debuginfod/Debuginfod.cpp",
    "llvm/include/llvm/Demangle/ItaniumDemangle.h",
    "llvm/lib/CodeGen/MIRParser/MIRParser.cpp",
    "llvm/lib/Support/ErrorHandling.cpp",
    "llvm/lib/Support/SourceMgr.cpp",
    "llvm/lib/Support/MemAlloc.cpp",
    "llvm/lib/Target/ARM/ARMExpandPseudoInsts.cpp",
    "llvm/include/llvm/IR/BasicBlock.h",
    "llvm/utils/TableGen/RegisterInfoEmitter.cpp",
    "llvm/utils/TableGen/CodeGenDAGPatterns.cpp",
    "llvm/utils/TableGen/CodeGenTarget.cpp",
    "llvm/utils/TableGen/DAGISelMatcher.cpp",
    "llvm/lib/Transforms/Vectorize/VPlanRecipes.cpp",
    "llvm/lib/Target/AArch64/AArch64ISelLowering.cpp",
    "llvm/include/llvm/CodeGen/MachineOutliner.h",
    "llvm/include/llvm/Analysis/RegionInfoImpl.h",
    "llvm/lib/Target/AArch64/Utils/AArch64SMEAttributes.cpp",
    "llvm/lib/CodeGen/TargetPassConfig.cpp",
    "llvm/include/llvm/CodeGen/TargetLowering.h",
    "llvm/lib/Target/AMDGPU/AMDGPUInstructionSelector.cpp",
    "llvm/lib/Transforms/Vectorize/VPlan.cpp",
    "llvm/lib/CodeGen/LiveRangeEdit.cpp",
    "llvm/lib/Transforms/Utils/ScalarEvolutionExpander.cpp",
    "llvm/lib/Transforms/Vectorize/VPlanTransforms.cpp",
    "llvm/lib/Analysis/TargetLibraryInfo.cpp",
    "llvm/utils/TableGen/SequenceToOffsetTable.h",
    "llvm/utils/TableGen/CodeGenDAGPatterns.cpp",
    "llvm/utils/TableGen/DAGISelMatcher.h",
    "llvm/utils/TableGen/CodeGenDAGPatterns.cpp",
    "llvm/utils/TableGen/CodeGenInstruction.h",
    "llvm/include/llvm/CodeGen/TargetInstrInfo.h",
    "llvm/lib/Target/X86/X86InstrFoldTables.cpp",
    "llvm/tools/dsymutil/DwarfLinkerForBinary.cpp",
    "llvm/lib/CodeGen/AssignmentTrackingAnalysis.cpp",
    "llvm/lib/Debuginfod/Debuginfod.cpp",
    "llvm/include/llvm/ADT/SetVector.h",
    "llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp",
    "llvm/lib/IR/DebugInfo.cpp",
    "llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp",
    "llvm/lib/Transforms/Utils/LoopUtils.cpp",
    "llvm/lib/Analysis/ConstantFolding.cpp",
    "llvm/lib/CodeGen/GlobalISel/CombinerHelper.cpp",
    "llvm/utils/TableGen/CodeGenRegisters.h",
    "llvm/lib/Analysis/ValueTracking.cpp",
    "llvm/utils/TableGen/CodeGenRegisters.cpp",
    "llvm/lib/Target/RISCV/RISCVFrameLowering.cpp",
    "llvm/utils/TableGen/CodeGenRegisters.h",
    "llvm/utils/TableGen/CodeGenRegisters.h",
    "llvm/utils/TableGen/CodeGenRegisters.h",
    "llvm/utils/TableGen/CodeGenRegisters.cpp",
    "llvm/lib/CodeGen/GlobalISel/CombinerHelper.cpp",
    "llvm/lib/Transforms/Vectorize/VPlanVerifier.cpp",
    "llvm/lib/CodeGen/GlobalISel/LegalizerHelper.cpp",
    "llvm/lib/Target/AArch64/AArch64FrameLowering.cpp",
    "llvm/lib/Target/AMDGPU/SIInstrInfo.cpp",
    "llvm/lib/ProfileData/InstrProf.cpp",
    "llvm/utils/TableGen/CodeGenDAGPatterns.cpp",
    "llvm/utils/TableGen/CodeGenDAGPatterns.cpp",
    "llvm/utils/TableGen/CodeGenRegisters.cpp",
    "llvm/utils/TableGen/CodeGenDAGPatterns.cpp",
    "llvm/utils/TableGen/RegisterInfoEmitter.cpp",
    "llvm/utils/TableGen/CodeGenTarget.cpp",
    "llvm/utils/TableGen/CodeGenDAGPatterns.cpp",
    "llvm/utils/TableGen/CodeGenDAGPatterns.cpp",
]

NEED_FILES_ONLY_NAME = (file.split('/')[-1] for file in NEED_FILES)


def split_instances(input_list: list, n: int) -> list:
    """
    Split a list into n approximately equal length sublists

    Args:
        input_list (list): List to split
        n (int): Number of sublists to split into
    Returns:
        result (list): List of sublists
    """
    avg_length = len(input_list) // n
    remainder = len(input_list) % n
    result, start = [], 0

    for i in range(n):
        length = avg_length + 1 if i < remainder else avg_length
        sublist = input_list[start: start + length]
        result.append(sublist)
        start += length

    return result


def is_valid_pull(pull: dict):
    for file in NEED_FILES:
        if file in pull["patch"]:
            return True

    for file in NEED_FILES_ONLY_NAME:
        if file in pull["patch"]:
            return True
    return False


def create_refactored_dataset(path_refactored: str, path_task: str):
    """
        Logic for creating refactored dataset with only diff operations with needed files

        Args:
            path_task (str): Path to get tasks instance data files
            path_refactored (str): Path to save new task instance data files to
        """
    repos = dict()
    completed = 0
    with_tests = 0
    total_instances = 0
    all_output = path_refactored + ".all"

    # Write to .all file for all PRs
    write_mode_all = "w" if not os.path.exists(all_output) else "a"
    with open(all_output, write_mode_all) as all_output:
        path_refactored1 = path_refactored
        write_mode = "w" if not os.path.exists(path_refactored1) else "a"
        with open(path_refactored1, write_mode) as path_refactored1:
            for ix, line in enumerate(open(path_task)):
                print(path_task)
                total_instances += 1
                pull = json.loads(line)
                if ix % 100 == 0:
                    logger.info(
                        f"(Up to {ix} checked) "
                        f"{completed} valid, {with_tests} with tests."
                    )
                if is_valid_pull(pull):
                    # If valid, write to .all and .jsonl output file
                    print(
                        json.dumps(pull), end="\n", flush=True, file=all_output
                    )  # write all instances to a separate file
                    print(
                        json.dumps(pull), end="\n", flush=True, file=path_refactored1
                    )  # write all instances to a separate file


def construct_data_files(data: dict):
    """
    Logic for combining multiple .all PR files into a single fine tuning dataset

    Args:
        data (dict): Dictionary containing the following keys:
            repos (list): List of repositories to retrieve instruction data for
            path_tasks (str): Path to get tasks instance data files
            path_refactored (str): Path to save new task instance data files to
            token (str): GitHub token to use for API requests
    """
    repos, path_tasks, path_refactored, max_pulls, cutoff_date, token = (
        data["repos"],
        data["path_tasks"],
        data["path_refactored"],
        data["max_pulls"],
        data["cutoff_date"],
        data["token"],
    )
    for repo in repos:
        repo = repo.strip(",").strip()
        repo_name = repo.split("/")[1]
        try:
            path_task = os.path.join(path_tasks, f"{repo_name}-task-instances.jsonl.all")
            print(path_task)
            path_refactored_one = os.path.join(path_refactored, f"{repo_name}-refactored-task-instances.jsonl")
            if not os.path.exists(path_task):
                print(
                    f"📁 Task instance data for {repo} is not exists at {path_task}, skipping..."
                )
            else:
                create_refactored_dataset(path_refactored_one, path_task)

        except Exception as e:
            print("-" * 80)
            print(f"Something went wrong for {repo}, skipping: {e}")
            print("Here is the full traceback:")
            traceback.print_exc()
            print("-" * 80)


def main(
        repos: list,
        path_tasks: str,
        path_refactored_tasks: str,
        max_pulls: int = None,
        cutoff_date: str = None,
):
    """
    Spawns multiple threads given multiple GitHub tokens for collecting fine tuning data

    Args:
        repos (list): List of repositories to retrieve instruction data for
        path_prs (str): Path to save PR data files to
        path_tasks (str): Path to save task instance data files to
        cutoff_date (str): Cutoff date for PRs to consider in format YYYYMMDD
    """
    path_tasks, path_refactored_tasks = os.path.abspath(path_tasks), os.path.abspath(path_refactored_tasks)
    print(f"Will save task instance data to {path_refactored_tasks}")
    print(f"Received following repos to create task instances for: {repos}")

    tokens = '.'

    tokens = tokens.split(",")
    data_task_lists = split_instances(repos, len(tokens))

    data_pooled = [
        {
            "repos": repos,
            "path_tasks": path_tasks,
            "path_refactored": path_refactored_tasks,
            "max_pulls": max_pulls,
            "cutoff_date": cutoff_date,
            "token": token,
        }
        for repos, token in zip(data_task_lists, tokens)
    ]

    with Pool(len(tokens)) as p:
        p.map(construct_data_files, data_pooled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos",
        nargs="+",
        help="List of repositories (e.g., `sqlfluff/sqlfluff`) to create task instances for",
    )
    parser.add_argument(
        "--path_tasks",
        type=str,
        help="Path to folder to get task instance data files",
    )
    parser.add_argument(
        "--path_refactored_tasks",
        type=str,
        help="Path to folder to save task instance data files to",
    )
    parser.add_argument(
        "--max_pulls", type=int, help="Maximum number of pulls to log", default=None
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        help="Cutoff date for PRs to consider in format YYYYMMDD",
        default=None,
    )
    args = parser.parse_args()
    main(**vars(args))
