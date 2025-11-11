    def _generate_visualization(self, result: AnalysisResultObject) -> go.Figure:
        """
        [결합 버전]
        VDB(df_base)를 배경으로, k-NN 결과와 동적 궤적을 함께 그립니다.       
        Args:
            result (AnalysisResultObject): EGO_compute로부터 반환된 *최신 C++ 분석 객체*.        
        Returns:
            go.Figure: 3D Plotly Figure object
        """

        fig = go.Figure()

        # --- 1. [VDB 시각화] 정적 데이터베이스 배경 (faded points) ---
        if not self.df_base.empty:
            fig.add_trace(go.Scatter3d(
                x=self.df_base['valence'], y=self.df_base['arousal'], z=self.df_base['dominance'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=self.df_base['random_val'], # 랜덤 색상
                    colorscale='hsv',
                    opacity=0.3, # 궤적이 잘 보이도록 0.3으로 설정
                    showscale=False
                ),
                text=self.df_base['term'],
                hoverinfo='text',
                name='VDB (All Emotions)'
            ))

        # --- 2. [VDB 시각화] k-NN 검색 결과 (green dots) ---
        if self.last_emotion and 'result' in self.last_emotion:
            results_list = []
            for item in self.last_emotion.get('result', []):
                results_list.append({
                    'valence': item['VAD']['V'],
                    'arousal': item['VAD']['A'],
                    'dominance': item['VAD']['D'],
                    'emotion': item['emotion'], 
                    'similarity': item.get('similarity_percent', 0)
                })
            df_results = pd.DataFrame(results_list)
            
            if not df_results.empty:
                fig.add_trace(go.Scatter3d(
                    x=df_results['valence'], y=df_results['arousal'], z=df_results['dominance'],
                    mode='markers',
                    marker=dict(size=6, color='limegreen'), # 녹색 점
                    text=df_results['emotion'],
                    customdata=df_results['similarity'], 
                    hovertemplate='<b>%{text}</b><br>Similarity: %{customdata:.2f}%', 
                    name='k-NN Results'
                ))

        # --- 3. [deltaEGO 시각화] 동적 궤적 (blue line) ---
        if self.emotion_history_VADPoint:
            v_coords = [p['v'] for p in self.emotion_history_VADPoint]
            a_coords = [p['a'] for p in self.emotion_history_VADPoint]
            d_coords = [p['d'] for p in self.emotion_history_VADPoint]
            
            fig.add_trace(go.Scatter3d(
                x=v_coords, y=a_coords, z=d_coords,
                mode='lines+markers',
                marker=dict(size=4, opacity=0.9, color='blue'),
                line=dict(width=4, color='blue'),
                name='VAD Trajectory'
            ))

        # --- 4. [deltaEGO 시각화] 현재 VAD (red diamond) ---
        if self.last_emotion_VADPoint:
            fig.add_trace(go.Scatter3d(
                x=[self.last_emotion_VADPoint['v']], 
                y=[self.last_emotion_VADPoint['a']], 
                z=[self.last_emotion_VADPoint['d']],
                mode='markers',
                marker=dict(symbol='diamond', size=8, color='red'),
                name='Current VAD'
            ))

        # --- 5. [deltaEGO 시각화] 구체 (Average & Stability) ---
        avg_area = result.cumulative.average_area
        if avg_area and avg_area.radius > 0:
            u, v_ = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
            x = avg_area.radius * np.outer(np.cos(u), np.sin(v_)) + avg_area.x
            y = avg_area.radius * np.outer(np.sin(u), np.sin(v_)) + avg_area.y
            z = avg_area.radius * np.outer(np.ones(np.size(u)), np.cos(v_)) + avg_area.z
            fig.add_trace(go.Surface(
                x=x, y=y, z=z, opacity=0.4,
                colorscale=[[0, 'green'], [1, 'lightgreen']],
                showscale=False, name='Average VAD Area',
            ))

        if self.default_axis and self.default_axis['baseline'] and self.default_axis['stabilityRadius'] > 0:
            baseline = self.default_axis['baseline']
            radius = self.default_axis['stabilityRadius']
            u, v_ = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
            x = radius * np.outer(np.cos(u), np.sin(v_)) + baseline['v']
            y = radius * np.outer(np.sin(u), np.sin(v_)) + baseline['a']
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v_)) + baseline['d']
            fig.add_trace(go.Surface(
                x=x, y=y, z=z, opacity=0.4,
                colorscale=[[0, 'orange'], [1, 'yellow']],
                showscale=False, name='Stability Radius',
            ))

        # --- 6. 최종 레이아웃 ---
        fig.update_layout(
            title=f"VAD 3D Trajectory ({self.ego_character} - Step {len(self.emotion_history_VADPoint)})",
            scene=dict(
                xaxis=dict(title="Valence", range=[-1, 1], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                yaxis=dict(title="Arousal", range=[-1, 1], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                zaxis=dict(title="Dominance", range=[-1, 1], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                aspectmode='cube' 
            ),
            # [FIX] 그래프 높이를 950으로 키웠습니다.
            height=950,
            showlegend=True,
            legend_title_text='Data Type'
        )
        
        return fig

# ==================================================================
# [테스트용 __main__ 블록]
# (이전의 test_visual.py와 동일한 기능)
# ==================================================================
if __name__ == "__main__":
    """
    이 스크립트가 직접 실행될 때 (python deltaEGO.py),
    결합된 VDB + Trajectory 시각화를 테스트합니다.
    """
    
    import sys
    from unittest import mock
    
    # 10단계 시나리오
    test_scenario = [
        {"name": "1. Calm", "V": 0.2, "A": -0.3, "D": 0.1},
        {"name": "2. Neutral", "V": 0.1, "A": 0.0, "D": 0.0},
        {"name": "3. Interested", "V": 0.4, "A": 0.2, "D": 0.3},
        {"name": "4. Alert", "V": 0.3, "A": 0.6, "D": 0.2},
        {"name": "5. Stressed!", "V": -0.4, "A": 0.7, "D": -0.5}, # <-- 스트레스
        {"name": "6. Coping", "V": -0.2, "A": 0.5, "D": -0.2},
        {"name": "7. Relieved", "V": 0.3, "A": 0.1, "D": 0.4},
        {"name": "8. Calming Down", "V": 0.4, "A": -0.2, "D": 0.3},
        {"name": "9. Satisfied", "V": 0.6, "A": 0.1, "D": 0.5}
    ]

    print("="*60)
    print("🚀 deltaEGO 결합 시각화 테스트 시작...")
    print("   Enter 키를 눌러 다음 단계의 감정을 적용하고 대시보드를 띄웁니다.")
    print("="*60)
    
    # --- 모의(Mock) 처리 ---
    # [수정] VDB 모듈 자체를 모의 처리하지 않고,
    # EGOSearcher 클래스만 모의 처리합니다. (df_base 로드는 필요하므로)
    
    mock_searcher_instance = mock.MagicMock()
    
    # VDB 모의 search 함수 정의 (k=5 반환)
    def mock_search_func(V, A, D, k, **kwargs):
        # VDB의 df_base를 사용하기 위해, deltaEGO 인스턴스를 통해 접근
        # (이 __main__에서는 deltaEGO 인스턴스가 먼저 생성되어야 함)
        # [수정] 인스턴스 접근이 어려우므로, 그냥 가짜 결과 반환
        return {
            "query": {"V": V, "A": A, "D": D, "k": k},
            "result": [
                {"emotion": "mocked_1", "VAD": {"V": V*0.9, "A": A*0.9, "D": D*0.9}, "similarity_percent": 90.0},
                {"emotion": "mocked_2", "VAD": {"V": V*0.8, "A": A*0.8, "D": D*0.8}, "similarity_percent": 80.0},
            ]
        }
    mock_searcher_instance.search = mock_search_func
    
    patcher = mock.patch(f"__main__.EGOSearcher", return_value=mock_searcher_instance)

    try:
        patcher.start()
        print("[1/2] 🔧 EGOSearcher (VDB 모듈) 모의 처리 완료.")
        
        ego = deltaEGO(character_name="CombinedVisualTester")
        
        # 만약 __init__에서 VDB 로딩에 실패했어도, 테스트는 진행
        if ego.df_base.empty:
            print("   [알림] VDB 배경 없이 궤적만 테스트합니다.")

        print("[2/2] 🧪 시나리오 테스트 시작...")
        
        start_time = time.time() - (len(test_scenario) * 10 * 60) 
        
        for i, step in enumerate(test_scenario):
            print(f"\n--- 📊 STEP {i+1}/{len(test_scenario)}: {step['name']} ---")
            print(f"    V={step['V']}, A={step['A']}, D={step['D']}")

            search_data = VAD_search(
                V=step['V'], A=step['A'], D=step['D'], k=5, dis=0.2
            )
            
            with mock.patch('time.time', return_value=start_time + i * 10 * 60):
                ego.VADsearch(search_data) # 모의 VDB 호출 (k-NN 결과 저장)
            
            print("\n    ... analize_VAD(visualize=True) 호출 중 ...")
            print("    ... 새 브라우저 탭에서 Plotly 대시보드를 확인하세요 ...")
            
            ego.analize_VAD(
                visualize=True, 
                return_analysis=False 
            )
            
            print(f"    ✅ Step {i+1} 시각화 생성 완료.")
            
            if i < len(test_scenario) - 1:
                try:
                    input("    Press Enter to apply the next emotion... (Ctrl+C to stop) ")
                except KeyboardInterrupt:
                    print("\n\nTest stopped by user.")
                    break
            else:
                print("\n" + "="*60)
                print("🎉 모든 시나리오 테스트 완료.")
                print("="*60)
                
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ [치명적 에러] 예상치 못한 오류 발생: {e}")
        print("="*60)
    finally:
        patcher.stop()