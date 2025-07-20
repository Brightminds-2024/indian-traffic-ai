import boto3
import json

REGION = "us-east-1"
INFERENCE_PROFILE_ARN = "arn:aws:bedrock:us-east-1:545009864608:inference-profile/us.meta.llama3-2-11b-instruct-v1:0"

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def get_ai_response(user_input):
    try:
        # ✅ Instructional prompt improves output quality
        formatted_prompt = (
            "You are a helpful AI assistant. Respond clearly and politely.\n\n"
            f"User: {user_input}\nAssistant:"
        )

        payload = {
            "prompt": formatted_prompt,
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = bedrock.invoke_model(
            modelId=INFERENCE_PROFILE_ARN,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )

        response_body = json.loads(response['body'].read())
        return response_body.get("generation", "No response received.")
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

