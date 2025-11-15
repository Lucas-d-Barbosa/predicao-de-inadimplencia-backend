from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def predict(request):
    
    if request.method == 'POST':
        
        try:
            data = json.loads(request.body) 
            print("Dados recebidos do frontend:", data)

        except json.JSONDecodeError:
            return JsonResponse({"error": "JSON inválido"}, status=400)

        mock_response = {
            "default": 0,
            "probability": 0.85
        }
        
        return JsonResponse(mock_response, status=200)
    
    else:
        return JsonResponse({"error": "Método não permitido. Use POST."}, status=405)