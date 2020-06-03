import requests

payload = '[{"temperature_00":7.1232254028,"wind_direction_00":103.17663,"wind_speed_00":8.133746,"temperature_08":6.4540021896,"wind_direction_08":107.79322,"wind_speed_08":6.326991,"temperature_16":7.2198843638,"wind_direction_16":119.070526,"wind_speed_16":3.062219},' \
          ' {"temperature_00":5.3762704213,"wind_direction_00":118.08433,"wind_speed_00":5.558247,"temperature_08":8.118838946,"wind_direction_08":116.193535,"wind_speed_08":8.565966,"temperature_16":9.3071762085,"wind_direction_16":120.26443,"wind_speed_16":11.993913},' \
          ' {"temperature_00":8.5934360504,"wind_direction_00":115.43259,"wind_speed_00":12.18185,"temperature_08":8.5879681269,"wind_direction_08":112.93136,"wind_speed_08":11.970859,"temperature_16":8.9567709605,"wind_direction_16":110.161095,"wind_speed_16":11.301485},' \
          ' {"temperature_00":8.0690326691,"wind_direction_00":103.169685,"wind_speed_00":9.983466,"temperature_08":7.9304853439,"wind_direction_08":106.04551,"wind_speed_08":6.3815556,"temperature_16":8.2289014181,"wind_direction_16":111.60216,"wind_speed_16":4.0873585}]'
headers = {'Content-Type': 'application/json; format=pandas-records'}
request_uri = 'http://127.0.0.1:5000/invocations'

if __name__ == '__main__':
   try:
      response = requests.post(request_uri, data=payload, headers=headers)
      print(response.content)
   except Exception as ex:s
      raise(ex)
