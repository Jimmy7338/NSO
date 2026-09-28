# V39 环境与临时存储记录

本轮因根分区不足，按用户授权仅迁移 NSO 的可再生运行缓存。未删除原始实验、已归档论文、训练权重、Git 对象或其他项目文件。宿主还有其他构建进程写盘，因此迁移前后的磁盘差值不能等同于本任务移动量。

1. `tmp/v38-tex` 的编译器、包缓存和临时编译文件，共 414 个文件、77,417,577 B，逐文件大小与 SHA-256 比对通过后迁至 `/dev/shm/nso_v39_tex_runtime`，原路径改为符号链接。
2. `.venv-3d/lib/python3.12/site-packages` 共 17,088 个文件、756,076,058 B，同样逐文件比对通过后迁至 `/dev/shm/nso_v39_python_packages`，原环境路径保留为符号链接。
3. 根分区又被其他构建写到约119 MiB余量时，将本轮不使用的 `.venv/lib/python3.12/site-packages/torch` 共10,856个文件、671,078,416 B按同样方式迁至 `/dev/shm/nso_v39_torch_cache`。模型权重没有移动。迁移后持久盘可用826,433,536 B、tmpfs余1,622,441,984 B；这只是当时资源快照。

完整清单见 `audit_results/v39_storage_20260920/`。这些目录是易失的运行缓存；持久保存的论文、源码、输入、传感包、地图、网格和复现收据均仍位于仓库磁盘。主机重启后需要恢复运行环境，不能将悬空符号链接误认为依赖仍在。

当前工具沙箱的 `/dev/shm` 是单独挂载，普通沙箱看不到宿主内存目录。此次 CPU 实验与测试均在已授权的宿主环境运行，固定 `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`，使用 Python `-B`。没有因为普通沙箱的导入失败重复安装依赖。宿主存在并行非本任务工作，耗时作为本机运行记录，不解释为独占机器上的严格速度排名。

恢复方式：空间充足时将对应缓存复制回原路径，并根据迁移清单校验；若主机已重启，可依据 `requirements-3d.lock.txt`、`requirements-3d-v23.txt` 和 V38 编译环境记录重建环境。Open3D CPU 0.19.0、NumPy 1.26.4、SciPy 1.11.4 为实际比较后端；新增 PDF 渲染工具为 pypdfium2 4.30.0。论文最终 PDF 和矢量图无需这些缓存即可阅读。

TARE 单独使用 `/dev/shm/nso_v39_tare`，ROS/PCL/Python3.8运行库恢复收据与再生脚本见 `V39_TARE_RUNTIME_PREFLIGHT_20260920.md`。不覆盖系统 Python、dpkg 或其他 ROS master。新物理任务执行前检查 64 MiB 持久磁盘余量；TARE 额外检查 2 GiB 系统可用内存和 512 MiB tmpfs 余量。
