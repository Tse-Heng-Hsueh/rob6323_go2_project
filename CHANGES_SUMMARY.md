# 修改摘要：133275 回归修復

**分支**: `fix-133275-regression`  
**日期**: 2025-12-15  
**目標**: 修復 133275 版本的性能回歸問題，回歸到接近 133131 的性能水平

---

## 🎯 修改概述

根據 Opus 模型的分析報告（`ANALYSIS_133275_regression.md` 和 `MODIFICATION_GUIDE.md`），我們識別並修復了以下關鍵問題：

1. ✅ **torque_limits 參數錯誤**：從 23.5 回滾到 Tutorial 推薦的 100.0
2. ✅ **暫時禁用 torque penalty**：確認 baseline 正常後再啟用
3. ✅ **重新實現 Part 6**：基於 IsaacGymEnvs 邏輯但使用 IsaacLab API

---

## 📝 詳細修改清單

### 1. rob6323_go2_env_cfg.py

#### 修改 1.1: 回滾 torque_limits 到 Tutorial 值
```python
# 修改前：
torque_limits = 23.5  # Max torque (realistic Go2 hardware limit)

# 修改後：
torque_limits = 100.0  # Max torque (Tutorial Part 2 recommended value)
```
**原因**: Tutorial Part 2 明確指定 100.0。23.5 限制過嚴，導致 PD 控制器輸出能力不足。

---

#### 修改 1.2: 暫時禁用 torque_reward_scale
```python
# 修改前：
torque_reward_scale = -0.0001  # Penalty for high torque usage (energy efficiency)

# 修改後：
# torque_reward_scale = -0.0001  # TODO: Re-enable after baseline validation (course requirement)
```
**原因**: Tutorial 未提及此參數，雖然課程要求有 torque penalty，但應先確認 baseline 正常後再單獨測試。

---

#### 修改 1.3: 移除 Tyler 專用參數並更新註釋
```python
# 修改前：
feet_target_clearance_height = 0.08
contact_force_scale = 50.0

# 修改後：
# Note: Implementation uses hardcoded values based on IsaacGymEnvs:
#   - feet clearance: 0.08 * phases + 0.02 (dynamic target height)
#   - contact force: 1 - exp(-F²/100) (squared force shaping)
```
**原因**: 新實現基於 IsaacGymEnvs，使用不同的常數值，不再需要這些配置參數。

---

### 2. rob6323_go2_env.py

#### 修改 2.1: 暫時禁用 torque buffer 和相關代碼
```python
# 註釋掉以下部分：
# - self._torques 初始化 (Line ~85)
# - "rew_torque" 在 episode_sums (Line ~103)
# - torque 儲存 (Line ~280)
# - rew_torque 計算 (Line ~409)
# - rewards 字典中的 rew_torque (Line ~424)
# - reward stack 中的 rew_torque (Line ~444)
```
**原因**: 與 cfg 修改一致，暫時禁用所有 torque penalty 相關代碼。

---

#### 修改 2.2: 重新實現 `_reward_feet_clearance()`

**關鍵改動**：
- ✅ 使用**動態 target_height**：`0.08 * phases + 0.02`
- ✅ phases 計算：`1.0 - torch.abs(1.0 - torch.clamp((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)`
- ✅ 在 swing 中期 target 最高（0.10m），開始/結束時較低（0.02m）
- ✅ 鼓勵自然的弧形足部軌跡

**與 Tyler 版本的差異**：
| 方面 | Tyler 版本 | 新實現（IsaacGymEnvs 邏輯） |
|------|-----------|---------------------------|
| Target height | 固定 0.08m | 動態 0.02-0.10m |
| 軌跡形狀 | 平坦 | 弧形（更自然） |
| 參數來源 | cfg.feet_target_clearance_height | Hardcoded |

---

#### 修改 2.3: 重新實現 `_reward_tracking_contacts_shaped_force()`

**關鍵改動**：
- ✅ **單向懲罰**：只懲罰 swing 時的接觸（不獎勵 stance）
- ✅ **力量塑形公式**：`1 - exp(-F²/100)` （F 的平方，不是線性）
- ✅ **平均化**：除以 4（4 隻腳）
- ✅ **返回負值**：懲罰導向

**與 Tyler 版本的差異**：
| 方面 | Tyler 版本 | 新實現（IsaacGymEnvs 邏輯） |
|------|-----------|---------------------------|
| 獎勵方向 | 雙向（stance+ swing-） | 單向（只有 swing-） |
| 力量塑形 | 1 - exp(-F/50) | 1 - exp(-F²/100) |
| 懲罰強度 | 較弱（線性） | 較強（平方） |

---

## 🔍 核心設計決策

### 為何採用 IsaacGymEnvs 的邏輯？

1. **經過驗證的算法**：IsaacGymEnvs 是 DMO 論文使用的代碼庫，已在真實 Go2 機器人上驗證
2. **更自然的足部運動**：動態 target height 鼓勵弧形軌跡，而非平坦抬腿
3. **更強的訓練信號**：F² 塑形對大力接觸有更強懲罰

### 為何不直接照抄 IsaacGymEnvs 代碼？

1. **API 差異**：IsaacGymEnvs 使用舊版 IsaacGym API，我們用 IsaacLab
2. **作業要求**：TA 明確要求 "reimplement/refactor"，不是照抄
3. **代碼質量**：我們保留了更好的註釋和 debug logging

---

## 📊 預期效果

根據 Opus 的分析，修改後應該看到：

| 指標 | 133275 當前值 | 預期目標值 | 133131 參考值 |
|------|--------------|-----------|--------------|
| track_lin_vel_xy_exp | ~22 | **~48** | 48.3 ✅ |
| track_ang_vel_z_exp | ~22 | **~24** | 24.4 |
| rew_action_rate | ~-5 | **~-2** | -2.2 ✅ |
| feet_clearance | ~0 | **~-0.7** | -0.75 ✅ |
| raibert_heuristic | ~-10 | **~-5** | -4.8 ✅ |

---

## 🚀 下一步行動

### 立即測試
```bash
# 在 HPC 上提交訓練任務
sbatch train.slurm
```

### 驗證檢查點
1. ✅ track_lin_vel_xy_exp 是否接近 48？
2. ✅ feet_clearance 是否有負值（不再是 ~0）？
3. ✅ rew_action_rate 是否改善（接近 -2）？
4. ✅ 訓練曲線是否穩定上升（不再先下降）？

### 如果效果良好
```python
# 解除註釋以下部分重新啟用 torque penalty：
# 1. cfg: torque_reward_scale = -0.0001
# 2. env: self._torques buffer
# 3. env: rew_torque 計算
# 4. env: rewards 字典中的 rew_torque
```

### 如果效果不佳
可能的調整方向：
1. 調整 feet_clearance_reward_scale（目前 -30.0）
2. 調整 tracking_contacts_shaped_force_reward_scale（目前 4.0）
3. 檢查 foot_indices 計算是否正確

---

## 📚 參考資源

- **Tutorial**: `tutorial/tutorial.md` (Parts 1-6)
- **IsaacGymEnvs 參考**: [go2_terrain.py](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/go2_terrain.py)
- **課程要求**: `rl_class_guidelines.md`
- **IsaacLab API**:
  - [ArticulationData](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData)
  - [ContactSensorData](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData)

---

## ⚠️ 重要提醒

1. **只修改兩個文件**：`rob6323_go2_env.py` 和 `rob6323_go2_env_cfg.py`
2. **base_height_min 保持 0.05**（用戶指定，不是 Tutorial 的 0.20）
3. **所有註釋使用英文**（符合專案規範）
4. **保留 debug logging**（便於診斷問題）

---

## 🔧 如何回滾（如果需要）

```bash
# 回到原始 133275 版本
git checkout main
git log --oneline  # 找到 133275 的 commit

# 或者回到 133239 版本（加入 sensor 前）
git checkout <133239-commit-hash>
```

