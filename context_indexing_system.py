#!/usr/bin/env python3
"""
Sistema de Indexación de Contexto para MCP
Mantiene memoria persistente y contexto del modelo entre sesiones
"""

import json
import time
import hashlib
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ContextIndexingSystem:
    """Sistema de indexación de contexto para memoria persistente del modelo"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or "mcp_context.db"
        self.session_id = self._generate_session_id()
        
        # Memoria en tiempo real
        self.current_context = {}
        self.conversation_buffer = deque(maxlen=100)  # Últimas 100 interacciones
        self.topic_tracker = defaultdict(list)
        
        # Configuración
        self.max_context_age_days = 30
        self.context_similarity_threshold = 0.7
        
        # Inicializar base de datos
        self._init_database()
        
        logger.info(f"🧠 Sistema de Indexación de Contexto iniciado")
        logger.info(f"📊 Sesión: {self.session_id}")
        logger.info(f"💾 Base de datos: {self.db_path}")
    
    def _generate_session_id(self) -> str:
        """Genera ID único de sesión"""
        timestamp = str(int(time.time()))
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def _init_database(self):
        """Inicializa la base de datos SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de contextos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                context_hash TEXT UNIQUE,
                topic TEXT,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                relevance_score REAL DEFAULT 1.0
            )
        ''')
        
        # Tabla de conversaciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                model_response TEXT,
                context_used TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tokens_used INTEGER,
                response_time REAL
            )
        ''')
        
        # Tabla de temas/proyectos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_name TEXT UNIQUE,
                description TEXT,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                interaction_count INTEGER DEFAULT 0
            )
        ''')
        
        # Índices para búsqueda rápida
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_context_hash ON contexts(context_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic ON contexts(topic)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Base de datos de contexto inicializada")
    
    def store_context(self, content: str, topic: str = "general", 
                     metadata: Dict[str, Any] = None) -> str:
        """Almacena contexto en el índice"""
        
        # Generar hash único del contenido
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Preparar metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'session_id': self.session_id,
            'stored_at': time.time(),
            'content_length': len(content)
        })
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insertar o actualizar contexto
            cursor.execute('''
                INSERT OR REPLACE INTO contexts 
                (session_id, context_hash, topic, content, metadata, last_accessed, access_count, relevance_score)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 
                        COALESCE((SELECT access_count FROM contexts WHERE context_hash = ?) + 1, 1),
                        ?)
            ''', (self.session_id, content_hash, topic, content, 
                  json.dumps(metadata), content_hash, 1.0))
            
            # Actualizar tabla de temas
            cursor.execute('''
                INSERT OR REPLACE INTO topics 
                (topic_name, description, keywords, last_updated, interaction_count)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP,
                        COALESCE((SELECT interaction_count FROM topics WHERE topic_name = ?) + 1, 1))
            ''', (topic, f"Contexto para {topic}", topic, topic))
            
            conn.commit()
            
            # Actualizar memoria en tiempo real
            self.current_context[content_hash] = {
                'content': content,
                'topic': topic,
                'metadata': metadata,
                'stored_at': time.time()
            }
            
            logger.info(f"💾 Contexto almacenado: {topic} ({len(content)} chars)")
            return content_hash
            
        except Exception as e:
            logger.error(f"Error almacenando contexto: {e}")
            return None
        finally:
            conn.close()
    
    def retrieve_context(self, query: str, topic: str = None, 
                        limit: int = 5) -> List[Dict[str, Any]]:
        """Recupera contexto relevante basado en query"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Construir query SQL
        sql_query = '''
            SELECT context_hash, topic, content, metadata, relevance_score, access_count
            FROM contexts 
            WHERE content LIKE ? 
        '''
        params = [f'%{query}%']
        
        if topic:
            sql_query += ' AND topic = ?'
            params.append(topic)
        
        # Ordenar por relevancia y frecuencia de acceso
        sql_query += '''
            ORDER BY relevance_score DESC, access_count DESC, last_accessed DESC
            LIMIT ?
        '''
        params.append(limit)
        
        cursor.execute(sql_query, params)
        results = cursor.fetchall()
        
        # Actualizar contadores de acceso
        for result in results:
            context_hash = result[0]
            cursor.execute('''
                UPDATE contexts 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE context_hash = ?
            ''', (context_hash,))
        
        conn.commit()
        conn.close()
        
        # Formatear resultados
        formatted_results = []
        for result in results:
            context_hash, topic, content, metadata_str, relevance_score, access_count = result
            
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except:
                metadata = {}
            
            formatted_results.append({
                'hash': context_hash,
                'topic': topic,
                'content': content,
                'metadata': metadata,
                'relevance_score': relevance_score,
                'access_count': access_count
            })
        
        logger.info(f"🔍 Recuperados {len(formatted_results)} contextos para: '{query[:50]}...'")
        return formatted_results
    
    def store_conversation(self, user_query: str, model_response: str, 
                          context_used: List[str] = None, tokens_used: int = 0,
                          response_time: float = 0.0):
        """Almacena conversación completa"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        context_used_str = json.dumps(context_used) if context_used else "[]"
        
        cursor.execute('''
            INSERT INTO conversations 
            (session_id, user_query, model_response, context_used, tokens_used, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (self.session_id, user_query, model_response, context_used_str, 
              tokens_used, response_time))
        
        conn.commit()
        conn.close()
        
        # Actualizar buffer en memoria
        self.conversation_buffer.append({
            'user_query': user_query,
            'model_response': model_response,
            'timestamp': time.time(),
            'context_used': context_used or []
        })
        
        logger.info(f"💬 Conversación almacenada ({tokens_used} tokens, {response_time:.2f}s)")
    
    def get_session_context(self, session_id: str = None) -> Dict[str, Any]:
        """Obtiene contexto completo de una sesión"""
        
        target_session = session_id or self.session_id
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obtener contextos de la sesión
        cursor.execute('''
            SELECT topic, COUNT(*) as count, AVG(relevance_score) as avg_relevance
            FROM contexts 
            WHERE session_id = ?
            GROUP BY topic
            ORDER BY count DESC
        ''', (target_session,))
        
        topics = cursor.fetchall()
        
        # Obtener conversaciones recientes
        cursor.execute('''
            SELECT user_query, model_response, timestamp, tokens_used
            FROM conversations 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (target_session,))
        
        recent_conversations = cursor.fetchall()
        
        conn.close()
        
        return {
            'session_id': target_session,
            'topics': [{'topic': t[0], 'count': t[1], 'avg_relevance': t[2]} for t in topics],
            'recent_conversations': [
                {
                    'user_query': c[0],
                    'model_response': c[1][:200] + "..." if len(c[1]) > 200 else c[1],
                    'timestamp': c[2],
                    'tokens_used': c[3]
                } for c in recent_conversations
            ],
            'total_contexts': len(self.current_context),
            'buffer_size': len(self.conversation_buffer)
        }
    
    def optimize_context_index(self):
        """Optimiza el índice de contexto"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Limpiar contextos antiguos
        cutoff_date = datetime.now() - timedelta(days=self.max_context_age_days)
        
        cursor.execute('''
            DELETE FROM contexts 
            WHERE last_accessed < ? AND access_count < 2
        ''', (cutoff_date,))
        
        deleted_contexts = cursor.rowcount
        
        # Actualizar scores de relevancia basado en frecuencia de acceso
        cursor.execute('''
            UPDATE contexts 
            SET relevance_score = CASE 
                WHEN access_count > 10 THEN 1.0
                WHEN access_count > 5 THEN 0.8
                WHEN access_count > 2 THEN 0.6
                ELSE 0.4
            END
        ''')
        
        # Vacuum para optimizar espacio
        cursor.execute('VACUUM')
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧹 Optimización completada: {deleted_contexts} contextos antiguos eliminados")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de contexto"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Estadísticas generales
        cursor.execute('SELECT COUNT(*) FROM contexts')
        total_contexts = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT topic) FROM contexts')
        unique_topics = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(relevance_score) FROM contexts')
        avg_relevance = cursor.fetchone()[0] or 0
        
        # Top temas
        cursor.execute('''
            SELECT topic, COUNT(*) as count 
            FROM contexts 
            GROUP BY topic 
            ORDER BY count DESC 
            LIMIT 5
        ''')
        top_topics = cursor.fetchall()
        
        # Estadísticas de sesión actual
        cursor.execute('''
            SELECT COUNT(*), AVG(tokens_used), AVG(response_time)
            FROM conversations 
            WHERE session_id = ?
        ''', (self.session_id,))
        
        session_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_contexts': total_contexts,
            'total_conversations': total_conversations,
            'unique_topics': unique_topics,
            'average_relevance': round(avg_relevance, 2),
            'top_topics': [{'topic': t[0], 'count': t[1]} for t in top_topics],
            'current_session': {
                'session_id': self.session_id,
                'conversations': session_stats[0] or 0,
                'avg_tokens': round(session_stats[1] or 0, 1),
                'avg_response_time': round(session_stats[2] or 0, 2)
            },
            'memory_buffer_size': len(self.conversation_buffer),
            'active_contexts': len(self.current_context)
        }
    
    def search_similar_contexts(self, content: str, similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """Busca contextos similares usando hash y contenido"""
        
        threshold = similarity_threshold or self.context_similarity_threshold
        
        # Buscar por palabras clave comunes
        words = set(content.lower().split())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT context_hash, topic, content, relevance_score FROM contexts')
        all_contexts = cursor.fetchall()
        
        similar_contexts = []
        
        for context in all_contexts:
            context_hash, topic, stored_content, relevance_score = context
            stored_words = set(stored_content.lower().split())
            
            # Calcular similitud Jaccard
            intersection = words.intersection(stored_words)
            union = words.union(stored_words)
            
            if union:
                similarity = len(intersection) / len(union)
                
                if similarity >= threshold:
                    similar_contexts.append({
                        'hash': context_hash,
                        'topic': topic,
                        'content': stored_content,
                        'similarity': similarity,
                        'relevance_score': relevance_score
                    })
        
        # Ordenar por similitud y relevancia
        similar_contexts.sort(key=lambda x: (x['similarity'], x['relevance_score']), reverse=True)
        
        conn.close()
        
        return similar_contexts[:10]  # Top 10 más similares
    
    def export_context_memory(self, output_file: str = None) -> str:
        """Exporta toda la memoria de contexto"""
        
        output_file = output_file or f"context_export_{self.session_id}.json"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Exportar todos los datos
        cursor.execute('SELECT * FROM contexts')
        contexts = cursor.fetchall()
        
        cursor.execute('SELECT * FROM conversations')
        conversations = cursor.fetchall()
        
        cursor.execute('SELECT * FROM topics')
        topics = cursor.fetchall()
        
        conn.close()
        
        export_data = {
            'export_timestamp': time.time(),
            'session_id': self.session_id,
            'contexts': contexts,
            'conversations': conversations,
            'topics': topics,
            'stats': self.get_context_stats()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📤 Memoria de contexto exportada a: {output_file}")
        return output_file

# Instancia global del sistema de indexación
context_indexer = ContextIndexingSystem()
