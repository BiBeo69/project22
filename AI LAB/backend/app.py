import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _require_api_key():
    """
    Kiểm tra đã có API key chưa.
    Nếu chưa có (ví dụ khi chạy qua WSGI mà không set env),
    trả về lỗi 500 để tránh gọi OpenAI mà không có key.
    """
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY is not configured on the server."}), 500
    return None


def call_openai_chat(messages, temperature=0.2, max_tokens=800):
    """Generic helper for text-only chat."""
    err = _require_api_key()
    if err:
        return None, err

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=40)
    except requests.RequestException as e:
        return None, (jsonify({"error": f"Error calling OpenAI: {e}"}), 502)

    if resp.status_code != 200:
        return None, (jsonify({
            "error": f"OpenAI returned status {resp.status_code}",
            "details": resp.text
        }), 502)

    data = resp.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None, (jsonify({"error": "Unexpected response format from OpenAI.", "raw": data}), 502)

    return answer, None


def call_openai_vision(image_bytes, system_prompt, user_instruction, temperature=0.2, max_tokens=800):
    """
    Helper for vision (image + text) using Chat Completions.
    We send a data URL (base64) to GPT as an image_url.
    """
    err = _require_api_key()
    if err:
        return None, err

    # Encode image as base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    # For simplicity we label as JPEG; using .jpg images is recommended.
    image_data_url = f"data:image/jpeg;base64,{image_b64}"

    # Chat Completions format: system content is string,
    # user content is an array of parts (text + image_url)
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_instruction,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                        # "detail": "high",  # you can enable this if you want higher-cost, higher-detail vision
                    },
                },
            ],
        },
    ]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        return None, (jsonify({"error": f"Error calling OpenAI (vision): {e}"}), 502)

    if resp.status_code != 200:
        return None, (jsonify({
            "error": f"OpenAI (vision) returned status {resp.status_code}",
            "details": resp.text
        }), 502)

    data = resp.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None, (jsonify({"error": "Unexpected response format from OpenAI (vision).", "raw": data}), 502)

    return answer, None


# ========= Vision endpoints =========

@app.route("/api/chromosome", methods=["POST"])
def chromosome_api():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded."}), 400

    image_bytes = file.read()

    system_prompt = (
        "You are Doctor Cal, an AI assistant role-playing as a friendly virtual doctor in cytogenetics. "
        "You speak in simple, conversational English with only a few basic medical terms. "
        "You must clearly say that you are an AI and not a real human doctor, and that this is for education only, "
        "not a clinical diagnosis and not treatment advice. "
        "Always sound warm and human, never robotic."
    )

    user_instruction = (
        "You are looking at a chromosome / karyotype spread image. "
        "For educational purposes only, internally decide which ONE of these categories best fits what you see:\n"
        "- Normal karyotype\n"
        "- Possible Down syndrome (Trisomy 21)\n"
        "- Possible other chromosomal abnormality\n"
        "- Cannot assess from this image\n\n"
        "However, do NOT output a numbered or bulleted list. "
        "Instead, answer like a friendly doctor speaking to a patient in plain English. "
        "Start your answer with a short greeting such as: \"Hi, I'm Doctor Cal, your virtual AI doctor.\" "
        "Then briefly say which category you think fits best, and explain why in a few short sentences using simple language. "
        "If the image does not look like a chromosome or karyotype image at all, respond in a gentle way, for example: "
        "\"Hi, I'm Doctor Cal. Hmm, this doesn't really look like a chromosome image. Could you upload a clearer karyotype image so I can take a proper look?\" "
        "At the end of every answer, clearly say that this is an educational, non-diagnostic interpretation only and must not be used for clinical decisions. "
        "Do not structure the answer as 1), 2), 3) and do not use bullet points."
    )

    answer, err = call_openai_vision(image_bytes, system_prompt, user_instruction)
    if err:
        return err
    return jsonify({"analysis": answer})


@app.route("/api/cancer-cell", methods=["POST"])
def cancer_cell_api():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded."}), 400

    image_bytes = file.read()

    system_prompt = (
        "You are Doctor Cal, an AI assistant role-playing as a friendly virtual doctor in histopathology. "
        "You speak in simple, conversational English with only a few basic medical terms. "
        "You must clearly say that you are an AI and not a real human doctor, and that this is for education only, "
        "not a clinical diagnosis and not treatment advice. "
        "Always sound warm and human, never robotic."
    )

    user_instruction = (
        "You are looking at a microscopic image of cells. For educational purposes only, internally choose ONE category:\n"
        "- Likely benign/reactive\n"
        "- Suspicious for malignancy\n"
        "- Highly concerning for malignancy\n"
        "- Not assessable\n\n"
        "Do NOT output this as a numbered list. "
        "Answer like a friendly doctor speaking to a patient. "
        "Start with something like: \"Hi, I'm Doctor Cal, your virtual AI doctor.\" "
        "Then in plain English, explain whether you think the cells look more benign, suspicious, or highly concerning for cancer, "
        "and why, using 3–5 short sentences. Use simple words and only a few necessary medical terms. "
        "If the image does not look like a microscopy or cytology image at all, respond gently, for example: "
        "\"Hi, I'm Doctor Cal. Hmm, this doesn't really look like a microscopic cell image. Could you upload a clearer microscope image so I can have a proper look?\" "
        "At the end, always say clearly that this is an educational, non-diagnostic interpretation only and cannot be used to make treatment decisions. "
        "Do not use bullet points or numbered steps in your final answer."
    )

    answer, err = call_openai_vision(image_bytes, system_prompt, user_instruction)
    if err:
        return err
    return jsonify({"analysis": answer})


@app.route("/api/chest-xray", methods=["POST"])
def chest_xray_api():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded."}), 400

    image_bytes = file.read()

    system_prompt = (
        "You are Doctor Cal, an AI assistant role-playing as a friendly virtual doctor in chest radiology. "
        "You speak in simple, conversational English with only a few basic medical terms. "
        "You must clearly say that you are an AI and not a real human doctor, and that this is for education only, "
        "not a clinical diagnosis and not treatment advice. "
        "Always sound warm and human, never robotic."
    )

    user_instruction = (
        "You are looking at a chest X-ray image. For educational purposes only, internally decide whether you think:\n"
        "- There is no obvious abnormality\n"
        "- There is a possible abnormality\n"
        "- The image is not assessable\n\n"
        "Do NOT output this as a list. "
        "Answer like a friendly doctor speaking to a patient. "
        "Start with something like: \"Hi, I'm Doctor Cal, your virtual AI doctor.\" "
        "Then briefly describe, in a few sentences, what you notice on the X-ray in plain English, "
        "and whether you think there might be an abnormality or not. "
        "If the image does not look like a chest X-ray at all, respond gently, for example: "
        "\"Hi, I'm Doctor Cal. Hmm, this doesn't really look like a chest X-ray. Could you upload a clearer chest X-ray image so I can take a proper look?\" "
        "At the end, always state clearly that this is only an educational explanation and not a radiology report or medical diagnosis, "
        "and that real X-rays must be interpreted by qualified doctors. "
        "Do not use bullet points or numbered steps in your final answer."
    )

    answer, err = call_openai_vision(image_bytes, system_prompt, user_instruction)
    if err:
        return err
    return jsonify({"analysis": answer})


# ========= Numeric / text endpoints =========

@app.route("/api/bmi-analysis", methods=["POST"])
def bmi_analysis_api():
    data = request.get_json(silent=True) or {}
    weight = data.get("weight")
    height_cm = data.get("height_cm")
    sex = data.get("sex")

    if weight is None or height_cm is None:
        return jsonify({"error": "Missing weight or height_cm."}), 400

    system_prompt = (
        "You are Doctor Cal, an AI assistant role-playing as a friendly virtual doctor. "
        "You speak in simple, conversational English with only a few basic medical terms. "
        "You must clearly say that you are an AI and not a real human doctor. "
        "You can calculate BMI and talk about general BMI categories, but this is only for education, "
        "not for diagnosis or treatment decisions. "
        "Always sound warm and human, never robotic."
    )

    user_prompt = (
        f"The user reports: weight = {weight} kg, height = {height_cm} cm, sex = {sex or 'not specified'}.\n"
        "Internally, calculate the BMI and think about whether it fits an Asian-oriented category such as underweight, "
        "normal, overweight, or obese.\n\n"
        "Then give your final answer as if you are speaking directly to the user:\n"
        "- Start by saying something like: \"Hi, I'm Doctor Cal, your virtual AI doctor.\" "
        "- Tell them what BMI you calculated and which category it roughly falls into, using simple language.\n"
        "- Give a short explanation about what that means and a few general lifestyle suggestions in normal sentences.\n"
        "- Do NOT use numbered lists or bullet points; just write 1–3 short paragraphs.\n"
        "- At the end, clearly say this is general information for learning only and not medical advice or a diagnosis."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer, err = call_openai_chat(messages)
    if err:
        return err
    return jsonify({"analysis": answer})


@app.route("/api/dose-x", methods=["POST"])
def dose_x_api():
    data = request.get_json(silent=True) or {}
    weight = data.get("weight")
    age = data.get("age")
    egfr = data.get("egfr")

    if weight is None or age is None:
        return jsonify({"error": "Missing weight or age."}), 400

    system_prompt = (
        "You are Doctor Cal, an AI assistant role-playing as a friendly virtual doctor in pharmacology. "
        "You speak in simple, conversational English with only a few basic medical terms. "
        "You must clearly say that you are an AI and not a real human doctor. "
        "You must NOT give any real medication name, real dose, frequency, or prescribing instructions. "
        "You may only talk in very general, qualitative terms, and you must emphasize that this is a fictional, "
        "educational example and not medical advice."
    )

    user_prompt = (
        f"Educational scenario. A hypothetical patient has weight = {weight} kg, age = {age} years, "
        f"and estimated kidney function (eGFR) = {egfr if egfr is not None else 'not provided'}.\n\n"
        "We have a fictional medication called 'Drug X'. "
        "Internally, think about how age, weight, and kidney function could influence dosing in very general terms.\n\n"
        "Then give your final answer in natural language, as Doctor Cal speaking directly to the user. "
        "Start with something like: \"Hi, I'm Doctor Cal, your virtual AI doctor.\" "
        "Explain in simple sentences how age, weight, and kidney function might broadly affect the dose of a medicine, "
        "without giving any specific numbers or real drug names. "
        "Do not use bullet points or numbered steps; just write as one or two short paragraphs. "
        "At the end, clearly say that this is a fictional educational example and must not be used to choose or adjust any real drug."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer, err = call_openai_chat(messages)
    if err:
        return err
    return jsonify({"analysis": answer})


@app.route("/api/cardiac-risk", methods=["POST"])
def cardiac_risk_api():
    data = request.get_json(silent=True) or {}
    age = data.get("age")
    sex = data.get("sex")
    sbp = data.get("sbp")
    smoker = data.get("smoker")
    diabetes = data.get("diabetes")
    chol = data.get("chol")

    if age is None or sbp is None:
        return jsonify({"error": "Missing age or systolic blood pressure."}), 400

    system_prompt = (
        "You are Doctor Cal, an AI assistant role-playing as a friendly virtual doctor in cardiovascular health. "
        "You speak in simple, conversational English with only a few basic medical terms. "
        "You must clearly say that you are an AI and not a real human doctor. "
        "You may talk about rough, qualitative cardiovascular risk (low / borderline / intermediate / high), "
        "but you must not claim to use an official risk calculator or give exact scores. "
        "This is educational only, not a clinical decision tool."
    )

    user_prompt = (
        f"Educational scenario. Person characteristics:\n"
        f"- Age: {age} years\n"
        f"- Sex: {sex}\n"
        f"- Systolic BP: {sbp} mmHg\n"
        f"- Smoker: {smoker}\n"
        f"- Diabetes: {diabetes}\n"
        f"- Total cholesterol: {chol if chol is not None else 'not provided'} mmol/L\n\n"
        "Internally, think about a rough, qualitative 10-year cardiovascular risk category "
        "(for example low, borderline, intermediate, or high) based on these factors.\n\n"
        "Then give your final answer in natural language as Doctor Cal talking to the user. "
        "Start with something like: \"Hi, I'm Doctor Cal, your virtual AI doctor.\" "
        "In a few sentences, say roughly whether their risk sounds lower or higher and which factors are pushing the risk up. "
        "Offer a few general lifestyle suggestions in plain English. "
        "Do not use bullet points or numbered steps; just write 1–3 short paragraphs. "
        "At the end, clearly say that this is only an approximate, educational explanation and not a real guideline-based risk score or medical advice."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer, err = call_openai_chat(messages)
    if err:
        return err
    return jsonify({"analysis": answer})


@app.route("/api/lab-blood", methods=["POST"])
def lab_blood_api():
    data = request.get_json(silent=True) or {}
    hb = data.get("hb")
    wbc = data.get("wbc")
    plt = data.get("plt")

    if hb is None or wbc is None or plt is None:
        return jsonify({"error": "Missing Hb, WBC or PLT."}), 400

    system_prompt = (
        "You are Doctor Cal, an AI assistant role-playing as a friendly virtual doctor in hematology. "
        "You speak in simple, conversational English with only a few basic medical terms. "
        "You must clearly say that you are an AI and not a real human doctor. "
        "You can talk about whether Hb, WBC, and platelets look low, normal, or high compared with typical adult ranges, "
        "but you must emphasize that reference ranges vary and you must not diagnose specific diseases or suggest treatments. "
        "This is educational only."
    )

    user_prompt = (
        f"Educational lab interpretation.\n"
        f"- Hemoglobin (Hb): {hb} g/dL\n"
        f"- White blood cells (WBC): {wbc} x10^9/L\n"
        f"- Platelets (PLT): {plt} x10^9/L\n\n"
        "Internally, think about whether each value looks low, normal, or high compared with typical adult reference ranges. "
        "Also think of a few general educational examples of what might cause low or high values.\n\n"
        "Then answer like Doctor Cal talking to the user. "
        "Start with something like: \"Hi, I'm Doctor Cal, your virtual AI doctor.\" "
        "Describe in simple words whether the hemoglobin, white cells and platelets look low, normal or high, "
        "and give a brief explanation in normal sentences. "
        "Do not use bullet points or numbered steps; just write 1–3 short paragraphs. "
        "Avoid naming specific diseases as a firm diagnosis; keep it general. "
        "At the end, clearly say that this is only an educational explanation and real lab results must be interpreted by a doctor who knows the full clinical picture."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer, err = call_openai_chat(messages)
    if err:
        return err
    return jsonify({"analysis": answer})


@app.route("/api/doctor-chat", methods=["POST"])
def doctor_chat_api():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing question."}), 400

    system_prompt = (
        "You are Doctor Cal, an AI assistant role-playing as a friendly virtual doctor. "
        "You speak in simple, conversational English with only a few basic medical terms. "
        "You must clearly say that you are an AI and not a real human doctor. "
        "You may explain medical concepts, common causes, and general approaches, "
        "but you must NOT provide a personal diagnosis, must NOT prescribe or adjust medications, "
        "and must NOT tell the user to start or stop any specific treatment. "
        "Always encourage users to see a licensed healthcare professional for individual care. "
        "Avoid bullet points and numbered lists; reply as if you are talking naturally to the user."
    )

    user_prompt = (
        "The user has asked the following medical question. "
        "Answer as Doctor Cal in warm, easy-to-understand English. "
        "Start your response with something like: \"Hi, I'm Doctor Cal, your virtual AI doctor.\" "
        "Explain the topic in a friendly way, using normal sentences, and keep the language simple. "
        "Do not use bullet points or numbered steps; just write a few short paragraphs. "
        "At the end, remind the user that this is general information only and they should see a real doctor for personal medical advice.\n\n"
        f"User question: {question}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer, err = call_openai_chat(messages, temperature=0.4, max_tokens=900)
    if err:
        return err
    return jsonify({"answer": answer})


if __name__ == "__main__":
    # Khi chạy LOCAL: nếu chưa có env, hỏi nhập key
    if not OPENAI_API_KEY:
        try:
            from getpass import getpass
            OPENAI_API_KEY = getpass("Enter your OpenAI API key (input hidden): ").strip()
        except EOFError:
            print("No API key provided. Set OPENAI_API_KEY environment variable or run again.")
            raise SystemExit(1)

    if not OPENAI_API_KEY:
        print("Empty API key. Exiting.")
        raise SystemExit(1)

    app.run(host="127.0.0.1", port=5000, debug=True)
