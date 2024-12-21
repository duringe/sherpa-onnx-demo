import time  # 导入时间模块，用于计算语音生成的耗时
import soundfile as sf  # 导入SoundFile库，用于保存生成的语音文件
import sherpa_onnx  # 导入Sherpa ONNX库，用于运行语音合成模型
import os  # 导入os库，用于路径操作


class TTS:
    def __init__(self, model_path,
                 tokens_path,
                 lexicon_path=None,
                 data_dir="",
                 dict_dir="",
                 provider="cpu",
                 num_threads=2,
                 debug=False):
        self.sample_rate = None  # 初始化采样率为None

        # 配置 TTS 模型的配置项
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model_path,  # 模型文件路径
                    lexicon=lexicon_path,  # 词典文件路径
                    data_dir=data_dir,  # 数据目录路径
                    dict_dir=dict_dir,  # 词典目录路径
                    tokens=tokens_path,  # token 文件路径
                ),
                provider=provider,  # 设置计算提供者（默认为CPU）
                debug=debug,  # 是否启用调试模式
                num_threads=num_threads,  # 设置线程数
            ),
            rule_fsts="",  # 用于规则的有限状态转移（FST）文件路径（空表示不使用）
            max_num_sentences=1,  # 最大句子数目，默认为1
        )

        # 验证配置是否正确
        if not tts_config.validate():
            raise ValueError("Please check your config")  # 配置不正确时抛出错误

        # 初始化TTS模型
        self.tts = sherpa_onnx.OfflineTts(tts_config)
        self.sample_rate = self.tts.sample_rate  # 获取模型的采样率

    # 日志打印方法
    def log(self, msg):
        print(msg)

    # 语音生成方法
    def generate(self, text, sid=1, speed=1.0, output_filename="./output/generated.wav"):
        self.log(f"Generating speech for text: {text} | speed: {speed} | speaker id: {sid}")  # 打印生成的文本、语速、说话人ID

        start_time = time.time()  # 记录开始时间
        audio = self.tts.generate(
            text,  # 要转换为语音的文本
            sid=sid,  # 说话人ID
            speed=speed,  # 语速
        )
        end_time = time.time()  # 记录结束时间

        # 检查音频生成是否成功
        if len(audio.samples) == 0:
            self.log("Error: No audio samples generated.")  # 如果没有生成音频，打印错误信息
            return

        elapsed_seconds = end_time - start_time  # 计算生成语音的耗时
        audio_duration = len(audio.samples) / audio.sample_rate  # 计算生成的音频时长
        real_time_factor = elapsed_seconds / audio_duration  # 计算实时因子（实际生成速度与音频播放速度的比值）

        # 打印生成统计信息
        self.log(f"Audio generated successfully. Duration: {audio_duration:.2f}s | "
                 f"Generation Time: {elapsed_seconds:.2f}s | Real-time factor: {real_time_factor:.2f}")

        # 创建输出文件夹（如果不存在）
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # 将生成的音频保存为WAV文件
        sf.write(
            output_filename,  # 输出文件名
            audio.samples,  # 音频样本数据
            samplerate=audio.sample_rate,  # 采样率
            subtype="PCM_16",  # 音频格式设置为16位PCM
        )

        self.log(f"Audio saved to {output_filename}")  # 打印保存的文件路径


# 主程序部分
if __name__ == "__main__":
    # 初始化TTS类，加载模型文件和配置文件
    tts = TTS(
        model_path=r"./model.onnx",  # 模型文件路径
        tokens_path=r"./tokens.txt",  # token文件路径
        lexicon_path=r"./lexicon.txt",  # 词典文件路径
        dict_dir=r"./dict"  # 词典目录路径
    )

    # 获取用户输入
    while True:
        print("\n请逐项输入：")

        # 输入说话人ID
        sid_input = input("请输入说话人ID (默认: 1): ")
        sid = 1  # 默认值
        if sid_input.strip():
            try:
                sid = int(sid_input)
            except ValueError:
                print("无效的说话人ID，使用默认值 1")

        # 输入语速
        speed_input = input("请输入语速 (默认: 1.0): ")
        speed = 1.0  # 默认值
        if speed_input.strip():
            try:
                speed = float(speed_input)
            except ValueError:
                print("无效的语速，使用默认值 1.0")

        # 输入文本内容
        text = input("请输入要合成的文本内容: ")
        if not text.strip():
            print("文本不能为空，请重新输入。")
            continue

        # 输入文件名
        filename_input = input("请输入保存的文件名 (默认: generated.wav): ")
        if not filename_input.strip():
            filename_input = "generated.wav"
        if not filename_input.endswith(".wav"):
            filename_input += ".wav"

        # 拼接文件保存路径
        output_filename = os.path.join("./output", filename_input)

        # 调用 TTS 生成语音
        tts.generate(text, sid=sid, speed=speed, output_filename=output_filename)

        # 是否继续
        continue_input = input("是否继续生成语音？(y/n): ").strip().lower()
        if continue_input != "y":
            break
