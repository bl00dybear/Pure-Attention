import numpy as np
import struct

def write_test_to_file(f, M, N, K, A, B, C, test_num):
    """Scrie un test în format binar în fișier"""
    # Scrie doar dimensiunile, fără text
    f.write(struct.pack('iii', M, N, K))
    
    # Scrie matricea A (M x N)
    for i in range(M):
        for j in range(N):
            f.write(struct.pack('f', A[i, j]))
    
    # Scrie matricea B (N x K)
    for i in range(N):
        for j in range(K):
            f.write(struct.pack('f', B[i, j]))
    
    # Scrie matricea C (M x K) - rezultatul așteptat
    for i in range(M):
        for j in range(K):
            f.write(struct.pack('f', C[i, j]))

def generate_tests():
    with open('matmul_tests.bin', 'wb') as f:
        test_num = 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(N, K).astype(np.float32)
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32) * 10
        B = np.random.randn(N, K).astype(np.float32) * 10
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(N, K).astype(np.float32)
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32) * 5
        B = np.random.randn(N, K).astype(np.float32) * 5
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32) * 2
        B = np.random.randn(N, K).astype(np.float32) * 2
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randint(-10, 10, (M, N)).astype(np.float32)
        B = np.random.randint(-10, 10, (N, K)).astype(np.float32)
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(N, K).astype(np.float32)
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32) * 0.1
        B = np.random.randn(N, K).astype(np.float32) * 0.1
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32) * 3
        B = np.random.randn(N, K).astype(np.float32) * 3
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32) * 0.01
        B = np.random.randn(N, K).astype(np.float32) * 0.01
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1

        print(f"Generez testul {test_num}...")
        M, N, K = 256, 128, 256
        A = np.random.randn(M, N).astype(np.float32) * 0.01
        B = np.random.randn(N, K).astype(np.float32) * 0.01
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1

        print(f"Generez testul {test_num}...")
        M, N, K = 2048*2, 4096*4, 4096*2
        A = np.random.randn(M, N).astype(np.float32) * 0.01
        B = np.random.randn(N, K).astype(np.float32) * 0.01
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1

        print(f"Generez testul {test_num}...")
        M, N, K = 5000, 10000, 5000
        A = np.random.randn(M, N).astype(np.float32) * 0.01
        B = np.random.randn(N, K).astype(np.float32) * 0.01
        C = A @ B
        write_test_to_file(f, M, N, K, A, B, C, test_num)
        test_num += 1
        
        f.write(struct.pack('i', test_num - 1))
    
    print(f"\nAm generat {test_num - 1} teste în fișierul 'matmul_tests.bin'")

if __name__ == "__main__":
    generate_tests()