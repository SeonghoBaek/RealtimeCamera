import redis

REDIS_SERVER = '10.100.1.150'
REDIS_PORT = 6379

try:
    rds = redis.StrictRedis(host=REDIS_SERVER, port=REDIS_PORT, db=0)

    rds.set('training', 'restart')

    redis_ready = True

except:
    redis_ready = False

