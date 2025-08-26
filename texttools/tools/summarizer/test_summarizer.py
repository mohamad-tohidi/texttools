from openai import OpenAI

from texttools.tools.summarizer.summarizer import Summarizer

client = OpenAI(
    base_url="http://185.208.182.157:9310/v1", api_key="hamta_T0k3n879875412"
)

tool = Summarizer(client=client, model="gemma-3", use_reason=True)
c = tool.summarize(
    "سلام دوستان حالتون چطوره میخواستم یک نکته ای رو مطرح کنم و اون هم این هست که با توجه با شرایط پیش آمده، دوستان ما توانایی حضور و مشارکت در جلسه فردا را نخواهند داشت بدلیل اتفاقات ناگواری که برای خانواده های ایشان رخ داده است. خوشحالیم که شما را انسان هایی صبور میبینیم که این مسائل را درک میکنند."
)
print(c)
