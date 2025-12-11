# SiliconFlow Toolkit

🎯 How to Use?

To configure OpenCode and Charm for SiliconFlow provider run: 

```bash
python3 install.py
```

Follow the instructions, enter your API key when prompted.

Verify: 

```bash
python3 ~/.config/check_siliconflow_perf.py
```

Create a cron job to keep models updated:

```bash
# Add to crontab -e
0 3 * * * python3 ~/.config/update_siliconflow_models.py >> ~/.siliconflow_update.log 2>&1
```

